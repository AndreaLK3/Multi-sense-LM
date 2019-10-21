import GetInputData.RetrieveInputData as RID
import PrepareGraphInput.PrepareInput as PI


def exe(do_reset=False):
    vocabulary_chunk = RID.continue_retrieving_data()
    PI.prepare(vocabulary_chunk)