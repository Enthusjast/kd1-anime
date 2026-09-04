from kd1_anime.resources import ResourceCoordinator


def test_resource_coordinator_enforces_process_wide_limits():
    resources = ResourceCoordinator(llm_limit=1, visual_llm_limit=1, slurm_limit=1)

    assert resources.llm.acquire(blocking=False) is True
    assert resources.llm.acquire(blocking=False) is False
    resources.llm.release()

    assert resources.visual_llm.acquire(blocking=False) is True
    assert resources.visual_llm.acquire(blocking=False) is False
    resources.visual_llm.release()

    assert resources.try_acquire_slurm() is True
    assert resources.try_acquire_slurm() is False
    resources.release_slurm()
    assert resources.try_acquire_slurm() is True
    resources.release_slurm()


def test_zero_slurm_limit_means_unbounded():
    resources = ResourceCoordinator(llm_limit=1, slurm_limit=0)

    assert resources.try_acquire_slurm() is True
    assert resources.try_acquire_slurm() is True
    resources.release_slurm()
    resources.release_slurm()


def test_existing_slurm_jobs_count_against_limit_until_released():
    resources = ResourceCoordinator(llm_limit=1, slurm_limit=1)

    resources.register_existing_slurm()
    assert resources.try_acquire_slurm() is False
    resources.release_slurm()
    assert resources.try_acquire_slurm() is True
    resources.release_slurm()


def test_visual_limit_defaults_to_main_llm_limit_for_compatibility():
    resources = ResourceCoordinator(llm_limit=2, slurm_limit=0)

    assert resources.visual_llm.acquire(blocking=False) is True
    assert resources.visual_llm.acquire(blocking=False) is True
    assert resources.visual_llm.acquire(blocking=False) is False
    resources.visual_llm.release()
    resources.visual_llm.release()


def test_rag_limit_is_independent_from_visual_limit():
    resources = ResourceCoordinator(llm_limit=1, visual_llm_limit=1, rag_limit=2, slurm_limit=0)

    assert resources.rag.acquire(blocking=False) is True
    assert resources.rag.acquire(blocking=False) is True
    assert resources.rag.acquire(blocking=False) is False
    resources.rag.release()
    resources.rag.release()
