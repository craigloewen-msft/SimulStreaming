#!/usr/bin/env python3
from whisper_streaming.whisper_online_main import *

import sys
import argparse
import os
import logging
import numpy as np
import asyncio

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


######### Server objects

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

class Connection:
    '''it wraps websocket connection'''
    PACKET_SIZE = 32000*5*60 # 5 minutes # was: 65536

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.last_line = ""

    async def send(self, line):
        '''it doesn't send the same line twice, because it was problematic in online-text-flow-events'''
        if line == self.last_line:
            return
        await self.websocket.send_text(line)
        self.last_line = line

    async def receive_lines(self):
        message = await self.websocket.receive_text()
        return message

    async def non_blocking_receive_audio(self):
        try:
            data = await self.websocket.receive_bytes()
            return data
        except WebSocketDisconnect:
            return None
        except Exception as e:
            logger.error(f"Error receiving audio: {e}")
            return None

import io
import soundfile

# wraps socket and ASR object, and serves one client connection. 
# next client should be served by a new instance of this object
class ServerProcessor:

    def __init__(self, c, online_asr_proc, min_chunk):
        logger.info("ServerProcessor init")
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk

        self.is_first = True

    async def receive_audio_chunk(self):
        # receive all audio that is available by this time
        # blocks operation if less than self.min_chunk seconds is available
        # unblocks if connection is closed or a chunk is available
        out = []
        minlimit = self.min_chunk*SAMPLING_RATE
        while sum(len(x) for x in out) < minlimit:
            raw_bytes = await self.connection.non_blocking_receive_audio()
            if not raw_bytes:
                break
#            print("received audio:",len(raw_bytes), "bytes", raw_bytes[:10])
            sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")
            audio, _ = librosa.load(sf,sr=SAMPLING_RATE,dtype=np.float32)
            out.append(audio)
        if not out:
            return None
        conc = np.concatenate(out)
        if self.is_first and len(conc) < minlimit:
            return None
        self.is_first = False
        return np.concatenate(out)

    async def send_result(self, iteration_output):
        # output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if iteration_output:
            message = "%1.0f %1.0f %s" % (iteration_output['start'] * 1000, iteration_output['end'] * 1000, iteration_output['text'])
            print(message, flush=True, file=sys.stderr)
            await self.connection.send(message)
        else:
            logger.debug("No text in this segment")

    async def process(self):
        # handle one client connection
        self.online_asr_proc.init()
        logger.info("ServerProcessor process started")
        while True:
            a = await self.receive_audio_chunk()
            if a is None:
                break
            self.online_asr_proc.insert_audio_chunk(a)
            o = self.online_asr_proc.process_iter()
            try:
                await self.send_result(o)
            except WebSocketDisconnect:
                logger.info("websocket disconnected -- connection closed?")
                break
            except Exception as e:
                logger.error(f"Error during processing: {e}")
                break

#        o = online.finish()  # this should be working
#        self.send_result(o)

# Global variables for ASR objects
asr_instance = None
online_instance = None
min_chunk_instance = None

def main_server(factory, add_args):
    '''
    factory: function that creates the ASR and online processor object from args and logger.  
            or in the default WhisperStreaming local agreement backends (not implemented but could be).
    add_args: add specific args for the backend
    '''
    global asr_instance, online_instance, min_chunk_instance
    
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()

    # server options
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=43007)
    parser.add_argument("--warmup-file", type=str, dest="warmup_file", 
            help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. "
            "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

    # options from whisper_online
    processor_args(parser)

    add_args(parser)

    args = parser.parse_args()

    set_logging(args,logger)

    # setting whisper object by args 

    asr, online = asr_factory(args, factory)
    asr_instance = asr
    online_instance = online
    
    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size
    
    min_chunk_instance = min_chunk

    # warm up the ASR because the very first transcribe takes more time than the others. 
    # Test results in https://github.com/ufal/whisper_streaming/pull/81
    msg = "Whisper is not warmed up. The first chunk processing may take longer."
    if args.warmup_file:
        if os.path.isfile(args.warmup_file):
            a = load_audio_chunk(args.warmup_file,0,1)
            asr.warmup(a)
            logger.info("Whisper is warmed up.")
        else:
            logger.critical("The warm up file is not available. "+msg)
            sys.exit(1)
    else:
        logger.warning(msg)

    # FastAPI app and WebSocket endpoint
    app = FastAPI()

    # Add CORS middleware to allow WebSocket connections from any origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def get():
        return HTMLResponse("""
        <html>
            <head><title>Whisper Streaming Server</title></head>
            <body>
                <h1>Whisper Streaming WebSocket Server</h1>
                <p>Connect to ws://{host}:{port}/ws for audio streaming</p>
            </body>
        </html>
        """.format(host=args.host, port=args.port))

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        logger.info(f'Connected to client on {websocket.client}')
        connection = Connection(websocket)
        # Create a new online processor instance for each connection
        from copy import deepcopy
        online_copy = asr_factory(args, factory)[1]
        proc = ServerProcessor(connection, online_copy, min_chunk_instance)
        try:
            await proc.process()
        except WebSocketDisconnect:
            logger.info('WebSocket disconnected')
        except Exception as e:
            logger.error(f'Error in websocket connection: {e}')
        finally:
            logger.info('Connection to client closed')

    # Run the server
    logger.info(f'Starting server on {args.host}:{args.port}')
    uvicorn.run(app, host=args.host, port=args.port)
    logger.info('Server terminated.')