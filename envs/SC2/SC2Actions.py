from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as raw_pb
from s2clientprotocol import common_pb2 as common_pb

import numpy as np

class Actions(object):
    def __init__(self):
        pass


    def random_move(self, unit, pos):
        unit_tag = unit.tag
        x = pos % 64
        y = pos / 64
        return raw_pb.ActionRawUnitCommand(
            ability_id=3674,  # Move
            unit_tags=[unit_tag],
            target_world_space_pos=common_pb.Point2D(x=x, y=y)
    )

    def random_attack(unit, max_x, max_y):
        try:
            unit_tag = unit.tag
            x = np.random.uniform(0, max_x)
            y = np.random.uniform(0, max_y)
            return raw_pb.ActionRawUnitCommand(
                ability_id=23,  # Move
                unit_tags=[unit_tag],
                target_world_space_pos=common_pb.Point2D(x=x, y=y)
            )
        except Exception as e:
            print(f"attack action error {e}")


    def move_left(unit, pos, default_move_speed=4):
        try:
            unit_tag = unit.tag
            x = unit.pos.x - default_move_speed
            y = unit.pos.y
            return raw_pb.ActionRawUnitCommand(
                ability_id=23,  # Move
                unit_tags=[unit_tag],
                target_world_space_pos=common_pb.Point2D(x=x, y=y)
            )
        except Exception as e:
            print(f"move left error {e}")

    def move_right(unit, pos, default_move_speed=4):
        unit_tag = unit.tag
        x = unit.pos.x + default_move_speed
        y = unit.pos.y
        return raw_pb.ActionRawUnitCommand(
            ability_id=23,  # Move
            unit_tags=[unit_tag],
            target_world_space_pos=common_pb.Point2D(x=x, y=y)
        )
    def move_up(unit, pos, default_move_speed=4):
        unit_tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y - default_move_speed
        return raw_pb.ActionRawUnitCommand(
            ability_id=23,  # Move
            unit_tags=[unit_tag],
            target_world_space_pos=common_pb.Point2D(x=x, y=y)
        )

    def move_down(unit, pos, default_move_speed=4):
        unit_tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y + default_move_speed
        return raw_pb.ActionRawUnitCommand(
            ability_id=23,  # Move
            unit_tags=[unit_tag],
            target_world_space_pos=common_pb.Point2D(x=x, y=y)
        )

    def attack_closest_enemy(unit, enemy_units):
        closest_enemy = None
        min_distance_sq = float("inf")

        for enemy in enemy_units:
            dx = unit.pos.x - enemy.pos.x
            dy = unit.pos.y - enemy.pos.y
            distance_sq = dx * dx + dy * dy

            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                closest_enemy = enemy

        if closest_enemy is not None:
            return raw_pb.ActionRawUnitCommand(
                ability_id=23,  # Attack
                unit_tags=[unit.tag],
                target_unit_tag=closest_enemy.tag
            )
        else:
            return None
