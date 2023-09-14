import uuid
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)

class RobotConfigChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
        self.message = None 

    def send_config(self, genome_length, genome, springyness, rotation):    
        msg = OutgoingMessage()
        msg.write_int32(genome_length)
        for val in genome:
            msg.write_int32(val)
        for val in rotation:
            msg.write_int32(val)
        for val in springyness:
            msg.write_float32(val)
        super().queue_message_to_send(msg)

    def on_message_received(self, msg: IncomingMessage) -> None:
        genome_length = msg.read_int32()
        genome = []
        for _ in range(genome_length):
            genome.append(msg.read_int32())
        self.message = genome