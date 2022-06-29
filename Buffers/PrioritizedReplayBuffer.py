from Buffers.ExperienceBuffer import ExperienceBuffer, ExperienceSamples

class PrioritizedReplayBuffer(ExperienceBuffer):

    def __init__(self, max_sz, input_dim, device):
        super().__init__(max_sz, input_dim, device)