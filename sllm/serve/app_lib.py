# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
from contextlib import asynccontextmanager
import asyncio

import ray
import ray.exceptions
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import StreamingResponse
import orjson

from sllm.serve.logger import init_logger

logger = init_logger(__name__)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Connect to the Ray cluster
        # ray.init()
        yield
        # Shutdown the Ray cluster
        ray.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    @app.post("/register")
    async def register_handler(request: Request):
        body = await request.json()

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )
        try:
            await controller.register.remote(body)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Cannot register model, please contact the administrator",
            )

        return {"status": "ok"}

    @app.post("/update")
    async def update_handler(request: Request):
        body = await request.json()
        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        logger.info(f"Received request to update model {model_name}")
        try:
            await controller.update.remote(model_name, body)
        except ray.exceptions.RayTaskError as e:
            raise HTTPException(status_code=400, detail=str(e.cause))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return {"status": f"updated model {model_name}"}

    @app.post("/delete")
    async def delete_model(request: Request):
        body = await request.json()

        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )
        lora_adapters = body.get("lora_adapters", None)

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        if lora_adapters is not None:
            logger.info(
                f"Received request to delete LoRA adapters {lora_adapters} on model {model_name}"
            )
            await controller.delete.remote(model_name, lora_adapters)
        else:
            logger.info(f"Received request to delete model {model_name}")
            await controller.delete.remote(model_name)

        return {"status": f"deleted model {model_name}"}

    async def inference_handler(request: Request, action: str):
        body = await request.json()
        model_name = body.get("model")
        logger.info(f"Received request for model {model_name}")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        request_router = ray.get_actor(model_name, namespace="models")
        logger.info(f"Got request router for {model_name}")

        result = request_router.inference.remote(body, action)
        return await result

    async def fine_tuning_handler(request: Request):
        body = await request.json()
        model_name = body.get("model")
        logger.info(f"Received request for model {model_name}")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        request_router = ray.get_actor(model_name, namespace="models")
        logger.info(f"Got request router for {model_name}")

        result = request_router.fine_tuning.remote(body)
        return await result

    @app.post("/v1/chat/completions")
    async def generate_handler(request: Request):
        body = await request.json()
        # If stream requested, return Server-Sent Events stream
        if body.get("stream"):
            model_name = body.get("model")
            if not model_name:
                raise HTTPException(
                    status_code=400, detail="Missing model_name in request body"
                )

            request_router = ray.get_actor(model_name, namespace="models")

            async def stream_results():
                stream_info = None
                try:
                    # Get streaming info from router (includes queue reference)
                    stream_info = await request_router.inference.remote(body, "generate")
                    
                    if not isinstance(stream_info, dict) or not stream_info.get("streaming"):
                        # Not a streaming response or error
                        yield "data: " + orjson.dumps(stream_info).decode("utf-8") + "\n\n"
                        return
                    
                    # Get the queue directly from the response
                    stream_queue = stream_info["queue"]
                    timeout = 600.0 # set to 10 minutes
                    
                    # Stream processing loop
                    while True:
                        try:
                            # sleep to avoid busy-waiting
                            await asyncio.sleep(0.0001)
                            # Get chunk from queue with timeout
                            chunk = await asyncio.wait_for(stream_queue.get_async(), timeout=timeout)
                            
                            # None signals end of stream
                            if chunk is None:
                                if "instance_id" in stream_info:
                                    try:
                                        await request_router.cleanup_streaming_request.remote(stream_info["instance_id"])
                                    except Exception as cleanup_error:
                                        logger.error(f"Error during router cleanup: {cleanup_error}")
                                break
                            
                            yield "data: " + orjson.dumps(chunk).decode("utf-8") + "\n\n"
                            
                        except asyncio.TimeoutError:
                            yield "data: " + orjson.dumps({"error": {"message": "Stream timeout"}}).decode("utf-8") + "\n\n"
                            # Also cleanup on timeout
                            if "instance_id" in stream_info:
                                try:
                                    await request_router.cleanup_streaming_request.remote(stream_info["instance_id"])
                                except Exception as cleanup_error:
                                    logger.error(f"Error during router cleanup: {cleanup_error}")
                            break
                        
                except Exception as e:
                    out = {"error": {"message": str(e)}}
                    yield "data: " + orjson.dumps(out).decode("utf-8") + "\n\n"
                finally:
                    # Clean up resources
                    if stream_info and "request_id" in stream_info:
                        logger.info(f"Streaming completed for request {stream_info['request_id']}")
                        try:
                            await request_router.cleanup_streaming_request.remote(stream_info["instance_id"])
                        except Exception as cleanup_error:
                            logger.error(f"Error during router cleanup: {cleanup_error}")
                
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_results(), media_type="text/event-stream; charset=utf-8")

        # Non-streaming path
        return await inference_handler(request, "generate")

    @app.post("/v1/embeddings")
    async def embeddings_handler(request: Request):
        return await inference_handler(request, "encode")

    @app.post("/fine-tuning")
    async def fine_tuning(request: Request):
        return await fine_tuning_handler(request)

    @app.get("/v1/models")
    async def get_models():
        logger.info("Attempting to retrieve the controller actor")
        try:
            controller = ray.get_actor("controller")
            if not controller:
                logger.error("Controller not initialized")
                raise HTTPException(
                    status_code=500, detail="Controller not initialized"
                )
            logger.info("Controller actor found")
            result = await controller.status.remote()
            logger.info("Controller status retrieved successfully")
            return result
        except Exception as e:
            logger.error(f"Error retrieving models: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve models"
            )

    return app
