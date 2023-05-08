import subprocess
import os
import asyncio
import sys
from contextlib import suppress

from aiohttp import ClientWebSocketResponse
from s2clientprotocol import sc2api_pb2 as sc_pb


class ProtocolError(Exception):

    @property
    def is_game_over_error(self) -> bool:
        return self.args[0] in ["['Game has already ended']", "['Not supported if game has already ended']"]


class ConnectionAlreadyClosed(ProtocolError):
    pass


class Protocol:

    def __init__(self, ws):
        """
        A class for communicating with an SCII application.
        :param ws: the websocket (type: aiohttp.ClientWebSocketResponse) used to communicate with a specific SCII app
        """
        assert ws
        self._ws: ClientWebSocketResponse = ws

    async def __request(self, request):
        try:
            await self._ws.send_bytes(request.SerializeToString())
        except TypeError as exc:
            raise ConnectionAlreadyClosed("Connection already closed.") from exc

        response = sc_pb.Response()
        try:
            response_bytes = await self._ws.receive_bytes()
        except TypeError as exc:
           raise ConnectionAlreadyClosed("Connection already closed.") from exc
        except asyncio.CancelledError:
            # If request is sent, the response must be received before reraising cancel
            try:
                await self._ws.receive_bytes()
            except asyncio.CancelledError:
                sys.exit(2)
            raise

        response.ParseFromString(response_bytes)
        return response

    async def _execute(self, **kwargs):
        assert len(kwargs) == 1, "Only one request allowed by the API"

        response = await self.__request(sc_pb.Request(**kwargs))

        new_status = Status(response.status)
        if new_status != self._status:
        self._status = new_status

        if response.error:
            raise ProtocolError(f"{response.error}")

        return response

    async def ping(self):
        result = await self._execute(ping=sc_pb.RequestPing())
        return result

    async def quit(self):
        with suppress(ConnectionAlreadyClosed, ConnectionResetError):
            await self._execute(quit=sc_pb.RequestQuit())
class SimpleSC2Env():
    def __init__(self):
        self.create = sc_pb.RequestCreateGame(local_map=sc_pb.LocalMap(map_path="Simple64"))
        self.create.player_setup.add(type=2, race=2)
        self.create.player_setup.add(type=1, race=1)
        #self.join = sc_pb.RequestJoinGame(race=1, options=sc_pb.InterfaceOptions(raw=True))
        self.join = sc_pb.RequestJoinGame(race=1)


class SC2Process():
    def __init__self(self):
        self.test = True

    def launch(self):
        exec_dir = r"C:\Program Files (x86)\StarCraft II\Support"
        nwd = r"C:\Users\cddal\projects\SCRMBL"
        os.chdir(exec_dir)

        exec_path = r"C:\Program Files (x86)\StarCraft II\Versions\Base90136\SC2.exe -listen 127.0.0.1 -p 8168 -windowwidth 1024 -windowheight 1024"
        self.process = subprocess.Popen(exec_path)
        #os.chdir(nwd)
        #return process


if __name__ == "__main__":
    sc2proc = SC2Process()
    sc2proc.launch()
    sc2env = SimpleSC2Env()
    print("Made it")
