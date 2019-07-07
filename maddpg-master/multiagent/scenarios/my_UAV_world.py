import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_landmarks = 2
        world.observing_range = 0.7
        world.min_corridor = 0.06
        world.collaborative = False
        num_adversaries = 1
        # add agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.07  # 0.03
            agent.done = False
            agent.adversary = True if i < num_adversaries else False
        # add landmarks
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.4  # np.random.uniform(0.1, 0.2)
            if i == (len(world.landmarks) - 1):
                landmark.size = 0.07  # 0.03
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.done = False
            if agent.adversary:
                agent.color = np.array([0.35, 0.35, 0.85])
            else:
                agent.color = np.array([0.15, 1.00, 0.40])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i == len(world.landmarks) - 1:
                landmark.color = np.array([0.21, 0.105, 0.30])
            else:
                landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            flag = 1
            while flag:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                temp1 = []
                temp2 = []
                temp1.append(np.sqrt(np.sum(np.square(world.agents[0].state.p_pos - landmark.state.p_pos))))
                temp2.append(world.agents[0].size + landmark.size + world.min_corridor)
                for j in range(0, i):
                    temp1.append(np.sqrt(np.sum(np.square(world.landmarks[j].state.p_pos - landmark.state.p_pos))))
                    temp2.append(world.landmarks[j].size + landmark.size + world.min_corridor)
                if min(np.array(temp1) - np.array(temp2)) > 0:
                    flag = 0
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def adversary_reward(self, agent, world):
        rew = 0
        agents = self.good_agents(world)
        vel = np.sqrt(np.sum(np.square(agent.state.p_vel)))
        if vel >= 1:
            rew -= 0
        if agent.collide:
            for a in world.landmarks[0:-1]:
                if self.is_collision(a, agent):
                    rew -= 8
            for a in agents:
                dis = np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))
                # if dis <= world.observing_range:
                rew -= dis
            for ag in agents:
                if self.is_collision(ag, agent):
                    rew += 10
        return rew

    def agent_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        vel = np.sqrt(np.sum(np.square(agent.state.p_vel)))
        if vel >= 1:
            rew -= 0
        l = world.landmarks[-1]
        adversaries = self.adversaries(world)
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        if min(dists) < l.size:
            rew += 10
        else:
            rew -= min(dists)
        '''for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)'''
        if agent.collide:
            for a in world.landmarks[0:-1]:
                if self.is_collision(a, agent):
                    rew -= 8
            for a in adversaries:
                # dis = np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))
                if self.is_collision(a, agent):
                    rew -= 10
                    # (world.observing_range - dis) * 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos_temp = []
        min_observable_landmark = np.min([4, len(world.landmarks)])
        for entity in world.landmarks:  # world.entities:
            distance = np.sqrt(np.sum(np.square([entity.state.p_pos - agent.state.p_pos])))
            if distance < world.observing_range and (not entity == world.landmarks[-1]):
                entity_pos_temp.append([np.append(entity.state.p_pos - agent.state.p_pos, entity.size), distance])
        entity_pos_temp.sort(key=lambda pos: pos[1])
        entity_pos_temp = entity_pos_temp[0:min_observable_landmark]
        entity_pos = [entity_pos_temp[i][0] for i in range(len(entity_pos_temp))]  # position
        for i in range(len(entity_pos_temp), min_observable_landmark):
            entity_pos.append([-1, -1, -1])
        # target obs
        target = world.landmarks[-1]
        entity_pos.append(np.append(target.state.p_pos - agent.state.p_pos, target.size))



        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

    def done(self, agent, world):
        agents = self.good_agents(world)
        target_landmark = world.landmarks[-1]
        if agent.adversary:
            dis = np.sqrt(np.sum(np.square(agent.state.p_pos - agents[0].state.p_pos)))
            if dis <= agent.size + agents[0].size:
                return True
        else:
            dis = np.sqrt(np.sum(np.square(agent.state.p_pos - target_landmark.state.p_pos)))
            if dis <= agent.size + target_landmark.size:
                return True
        return False


    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]
