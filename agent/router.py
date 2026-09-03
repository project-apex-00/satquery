"""
router.py

The "agent" -- decides what kind of task the user's question is,
and which specialist model to call. Kept as simple keyword-based
logic on purpose: it's easy to explain to judges as "our own logic"
instead of a heavy framework, and it's honest about what it does.

NOTE: only the single-image classification path is wired up right now.
change_detection and fusion are placeholders until those pieces are built.
"""

from enum import Enum
from agent.audit_log import log_step


class TaskType(str, Enum):
    SINGLE_IMAGE_VQA = "single_image_vqa"
    CHANGE_DETECTION = "change_detection"      # not implemented yet
    OPTICAL_SAR_FUSION = "optical_sar_fusion"  # not implemented yet
    UNKNOWN = "unknown"


def classify_task(question: str, num_images: int) -> TaskType:
    q = question.lower()

    if num_images >= 2 and any(word in q for word in ["change", "differ", "between", "before", "after"]):
        task = TaskType.CHANGE_DETECTION
    elif any(word in q for word in ["sar", "radar", "fusion", "combine"]):
        task = TaskType.OPTICAL_SAR_FUSION
    elif num_images == 1:
        task = TaskType.SINGLE_IMAGE_VQA
    else:
        task = TaskType.UNKNOWN

    log_step("router_decision", {
        "question": question,
        "num_images": num_images,
        "chosen_task": task.value,
    })
    return task
