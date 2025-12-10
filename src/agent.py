import logging
import threading
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    inference,
    metrics,
)
from livekit.plugins import noise_cancellation, sarvam, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        base_instructions = (
            "Greet user as his name is known from the profile,like hii <name>! I am Talkypie, How can I assist you today?."
            "Dont take name of user multiple times in a single response. "
            "You are a kid assistant, who helps engage kids in a fun playful manner. "
            "Please be concise in your responses. Use very simple language that kids can understand and use short sentences. "
            "Your responses are concise, to the point, and without emojis or symbols."
        )

        profile_instructions = ""
        try:
            profile_path = os.path.join(os.getcwd(), "profile.json")
            if os.path.exists(profile_path):
                with open(profile_path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                parts = []
                if profile.get("name"):
                    parts.append(f"Child name: {profile['name']}.")
                if profile.get("age"):
                    parts.append(f"Age: {profile['age']}.")
                if profile.get("gender"):
                    parts.append(f"Gender: {profile['gender']}.")
                if profile.get("likes"):
                    parts.append(f"Likes: {profile['likes']}.")
                if profile.get("learning"):
                    parts.append(f"Current learning: {profile['learning']}")

                if parts:
                    profile_instructions = (
                        "Use this profile to tailor responses: " + " ".join(parts)
                    )
        except Exception:
            pass

        final_instructions = base_instructions + " " + profile_instructions
        super().__init__(instructions=final_instructions)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=sarvam.STT(language="hi-IN", model="saarika:v2.5"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=sarvam.TTS(target_language_code="hi-IN", speaker="manisha"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await ctx.connect()


def _start_profile_server():
    host = "0.0.0.0"
    port = int(os.getenv("PORT", "8080"))  # <-- required for Railway

    class Handler(BaseHTTPRequestHandler):
        def _headers(self, code=200, type="application/json"):
            self.send_response(code)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Type", type)
            self.end_headers()

        def do_OPTIONS(self):
            self._headers(204)

        def do_GET(self):
            if self.path == "/":
                self._headers(200, "text/plain")
                self.wfile.write(b"OK")
                return

            if self.path == "/profile":
                try:
                    path = os.path.join(os.getcwd(), "profile.json")
                    if not os.path.exists(path):
                        self._headers(204)
                        return

                    with open(path, "r", encoding="utf-8") as f:
                        data = f.read()
                    self._headers(200, "application/json")
                    self.wfile.write(data.encode("utf-8"))
                except Exception:
                    self._headers(500)
                return

            self._headers(404)

        def do_POST(self):
            if self.path != "/profile":
                self._headers(404)
                return

            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                data = json.loads(body.decode("utf-8"))

                with open("profile.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                self._headers(200)
                self.wfile.write(b"{}")
            except Exception:
                self._headers(500)

    server = HTTPServer((host, port), Handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Profile server running on {host}:{port}")


if __name__ == "__main__":
    _start_profile_server()
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))



# import logging
# import threading
# import json
# import os
# from http.server import BaseHTTPRequestHandler, HTTPServer

# from dotenv import load_dotenv
# from livekit.agents import (
#     Agent,
#     AgentSession,
#     JobContext,
#     JobProcess,
#     MetricsCollectedEvent,
#     RoomInputOptions,
#     WorkerOptions,
#     cli,
#     inference,
#     metrics,
# )
# from livekit.plugins import noise_cancellation, sarvam, silero
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

# logger = logging.getLogger("agent")

# # Load environment variables from .env.local
# load_dotenv(".env.local")


# class Assistant(Agent):
#     def __init__(self) -> None:
#         # Base instructions
#         base_instructions = (
#             "Greet user as his name is known from the profile,like hii <name>! I am Talkypie, How can I assist you today?."
#             "Dont take name of user multiple times in a single response. "
#             "You are a kid assistant, who helps engage kids in a fun playful manner. "
#             "Please be concise in your responses. Use very simple language that kids can understand and use short sentences. "
#             "You eagerly assist users with their questions by providing information from your extensive knowledge. "
#             "Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, "
#             "asterisks, or other symbols. You are curious, friendly, and have a sense of humor."
#         )

#         # If a profile file exists (written by the frontend via the profile endpoint),
#         # include it in the assistant instructions
#         profile_instructions = ""
#         try:
#             profile_path = os.path.join(os.getcwd(), "profile.json")
#             if os.path.exists(profile_path):
#                 with open(profile_path, "r", encoding="utf-8") as f:
#                     profile = json.load(f)
#                 parts = []
#                 name = profile.get("name")
#                 if name:
#                     parts.append(f"Child name: {name}.")
#                 age = profile.get("age")
#                 if age:
#                     parts.append(f"Age: {age}.")
#                 gender = profile.get("gender")
#                 if gender:
#                     parts.append(f"Gender: {gender}.")
#                 likes = profile.get("likes")
#                 if likes:
#                     parts.append(f"Likes/Dislikes: {likes}")
#                 learning = profile.get("learning")
#                 if learning:
#                     parts.append(f"Current learning: {learning}")

#                 if parts:
#                     profile_instructions = (
#                         "Use the following parent-provided profile to tailor your responses for the child: "
#                         + " ".join(parts)
#                     )
#         except Exception:
#             # best-effort only; don't crash the agent if profile can't be read
#             profile_instructions = ""

#         final_instructions = base_instructions + (
#             " " + profile_instructions if profile_instructions else ""
#         )

#         super().__init__(
#             instructions=final_instructions,
#         )


# def prewarm(proc: JobProcess):
#     # Preload Silero VAD once per worker process
#     proc.userdata["vad"] = silero.VAD.load()


# async def entrypoint(ctx: JobContext):
#     # Logging setup
#     ctx.log_context_fields = {
#         "room": ctx.room.name,
#     }

#     # Voice AI pipeline using Sarvam STT/TTS + OpenAI LLM + LiveKit turn detector + Silero VAD
#     session = AgentSession(
#         # STT: ears
#         stt=sarvam.STT(
#             language="hi-IN",
#             model="saarika:v2.5",
#         ),
#         # LLM: brain
#         llm=inference.LLM(model="openai/gpt-4.1-mini"),
#         # TTS: voice
#         tts=sarvam.TTS(
#             target_language_code="hi-IN",
#             speaker="manisha",
#         ),
#         # Turn detection model + VAD
#         turn_detection=MultilingualModel(),
#         vad=ctx.proc.userdata["vad"],
#         # Preemptive generation improves responsiveness
#         preemptive_generation=True,
#     )

#     usage_collector = metrics.UsageCollector()

#     @session.on("metrics_collected")
#     def _on_metrics_collected(ev: MetricsCollectedEvent):
#         metrics.log_metrics(ev.metrics)
#         usage_collector.collect(ev.metrics)

#     async def log_usage():
#         summary = usage_collector.get_summary()
#         logger.info(f"Usage: {summary}")

#     ctx.add_shutdown_callback(log_usage)

#     await session.start(
#         agent=Assistant(),
#         room=ctx.room,
#         room_input_options=RoomInputOptions(
#             noise_cancellation=noise_cancellation.BVC(),
#         ),
#     )

#     # Join the room and connect to the user
#     await ctx.connect()


# def _start_profile_server(
#     host: str = "0.0.0.0",
#     port: int = 8080,
# ):
#     """
#     Start a tiny HTTP server that accepts POST /profile to save a JSON profile to disk,
#     and GET /profile to read it.

#     This is intentionally dependency-free and minimal. The server responds with CORS headers
#     so the frontend (e.g. localhost:3000 or your Next.js app) can POST from the browser.
#     """

#     class Handler(BaseHTTPRequestHandler):
#         def _set_cors_headers(self):
#             self.send_header("Access-Control-Allow-Origin", "*")
#             self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
#             self.send_header("Access-Control-Allow-Headers", "Content-Type")

#         def do_OPTIONS(self):
#             self.send_response(204)
#             self._set_cors_headers()
#             self.end_headers()

#         def do_POST(self):
#             if self.path != "/profile":
#                 self.send_response(404)
#                 self._set_cors_headers()
#                 self.end_headers()
#                 return

#             content_length = int(self.headers.get("Content-Length", 0))
#             body = self.rfile.read(content_length) if content_length > 0 else b""
#             try:
#                 data = json.loads(body.decode("utf-8")) if body else {}
#             except Exception:
#                 self.send_response(400)
#                 self._set_cors_headers()
#                 self.end_headers()
#                 return

#             try:
#                 profile_path = os.path.join(os.getcwd(), "profile.json")
#                 with open(profile_path, "w", encoding="utf-8") as f:
#                     json.dump(data, f, ensure_ascii=False, indent=2)
#                 self.send_response(200)
#                 self._set_cors_headers()
#                 self.send_header("Content-Type", "application/json")
#                 self.end_headers()
#                 self.wfile.write(b"{}")
#             except Exception:
#                 self.send_response(500)
#                 self._set_cors_headers()
#                 self.end_headers()

#         def do_GET(self):
#             # Serve saved profile.json for client-side auto-start
#             if self.path == "/":
#                 # Basic health-check endpoint for Render (HEAD/GET /)
#                 self.send_response(200)
#                 self._set_cors_headers()
#                 self.send_header("Content-Type", "text/plain")
#                 self.end_headers()
#                 self.wfile.write(b"OK")
#                 return

#             if self.path != "/profile":
#                 self.send_response(404)
#                 self._set_cors_headers()
#                 self.end_headers()
#                 return

#             try:
#                 profile_path = os.path.join(os.getcwd(), "profile.json")
#                 if not os.path.exists(profile_path):
#                     # No profile saved yet
#                     self.send_response(204)
#                     self._set_cors_headers()
#                     self.end_headers()
#                     return

#                 with open(profile_path, "r", encoding="utf-8") as f:
#                     data = f.read()

#                 self.send_response(200)
#                 self._set_cors_headers()
#                 self.send_header("Content-Type", "application/json")
#                 self.end_headers()
#                 self.wfile.write(data.encode("utf-8"))
#             except Exception:
#                 self.send_response(500)
#                 self._set_cors_headers()
#                 self.end_headers()

#         def do_HEAD(self):
#             # Health check for HEAD /
#             if self.path == "/":
#                 self.send_response(200)
#                 self._set_cors_headers()
#                 self.end_headers()
#             else:
#                 self.send_response(404)
#                 self._set_cors_headers()
#                 self.end_headers()

#     try:
#         server = HTTPServer((host, port), Handler)
#     except Exception:
#         logger.exception("Profile server failed to start on %s:%s", host, port)
#         return

#     def serve():
#         logger.info("Profile endpoint listening on http://%s:%s/profile", host, port)
#         try:
#             server.serve_forever()
#         except Exception:
#             pass

#     thread = threading.Thread(target=serve, daemon=True)
#     thread.start()


# if __name__ == "__main__":
#     # Start profile server ONLY in the main process, to avoid port conflicts
#     # and inference timeouts. Render will also use this port (8080) for health checks.
#     try:
#         # Allow overriding port via env if you want later
#         profile_port = int(os.getenv("PROFILE_PORT", "8080"))
#         _start_profile_server(host="0.0.0.0", port=profile_port)
#     except Exception:
#         logger.exception("Failed to initialize profile HTTP endpoint")

#     # Start LiveKit worker as before
#     cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))