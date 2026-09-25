# Assignment 2 AI Usage Log

## Purpose and scope

This log documents the use of AI assistance while preparing Assignment 2, *Learning from Click-Logs on EB-NeRD and MIND*. It is a transparency record for the submitted repository. The team remains responsible for checking every implementation choice, running the experiments, interpreting results, and making the final submission.

## Tool used

- **AI assistant:** Codex / ChatGPT coding assistant.
- **Use:** code explanation, implementation assistance, debugging, optimisation, documentation, design-note drafting, test creation, Git hygiene, and repository organisation.
- **Not used for:** fabricating dataset labels, leaderboard scores, or experimental measurements. Codabench submissions and the displayed leaderboard evidence were performed and verified by the team.

## Prompt and conversation index

The original Codex/ChatGPT conversation is the authoritative full chat history. Export that conversation from the chat interface and submit it with the Moodle deliverables if the instructor requires the raw transcript. This repository log indexes the substantive prompts that affected Assignment 2.

| Work item | Representative user request | AI assistance recorded |
| --- | --- | --- |
| Assignment interpretation | Read the Assignment 2 brief and check what is required for Q1-Q5, the report, and Codabench. | Explained the requirement-to-code mapping and identified deliverables. |
| Data pipeline and debugging | Fix the feature-store path and dependency issues while running the A2 pipeline. | Diagnosed missing archive paths and the macOS LightGBM/OpenMP issue; proposed and implemented scoped fixes. |
| Behavioural features and re-ranking | Implement click history, recency, session, popularity, freshness, causal feature boundaries, and a LambdaRank re-ranker. | Drafted or edited feature, re-ranker, ablation, evaluation, and leakage-test code. |
| Submission generation | Prepare MIND and EB-NeRD Codabench prediction ZIP files and make large-test generation memory efficient. | Drafted or edited streaming/checkpointed submission scripts and validation tests. |
| Experiment review | Improve the ranking pipeline, compare BM25 and semantic retrieval, and report bootstrap confidence intervals. | Helped analyse local metrics, create ablation code, and distinguish offline metrics from hidden-test leaderboard metrics. |
| Design note | Create the Assignment 2 design note, include large-test Codabench evidence, and correct the MIND metric distinction. | Drafted and edited the report source and PDF; incorporated user-provided screenshots and clearly labelled pending EB-NeRD scoring. |
| Repository delivery | Create the `sbidisha_A2` branch, remove Assignment 1 from that branch, consolidate the README, and prevent large artifacts from being committed. | Performed requested Git operations and edited repository documentation and ignore rules. |
| Final compliance check | Compare the repository against A2.pdf and improve Git hygiene. | Audited requirements, ran local tests, added this log, and excluded generated logs. |

## Authorship and review markings

| Artifact or activity | Marking | Team responsibility |
| --- | --- | --- |
| `features.py`, `reranker.py`, `ablation.py`, `offline_evaluation.py`, `serving_scale.py`, `serving_availability_ablation.py` | AI-assisted implementation and review | Verify feature definitions, causal boundaries, metrics, and all reported claims. |
| `build_pipeline.py`, `bm25_retrieval.py`, `semantic_retrieval.py`, `mind_large_v2.py`, `ebnerd_large_v2.py` | AI-assisted implementation and optimisation | Verify dataset parsing, temporal splits, retrieval behaviour, and reproducibility. |
| `make_mind_submission.py`, `make_ebnerd_personalized_submission.py`, and submission tests | AI-assisted implementation and debugging | Run, inspect, and upload only validated prediction archives. |
| `run_a2_pipeline.sh`, tests, README, and `create_mind_design_note_pdf.py` | AI-assisted documentation, automation, and test support | Review commands, report wording, and final repository contents. |
| Dataset downloads, Codabench registration, archive execution, submission uploads, and screenshots | Human-operated | Retain original evidence and do not alter observed scores or statuses. |
| Experimental JSON outputs and leaderboard values | Runtime- or platform-generated evidence | Report only values produced by the local code or shown by Codabench. |

## Human review checklist before submission

- [ ] Export the complete chat history and attach it to the Moodle submission if required.
- [ ] Verify that each listed AI-assistance label accurately reflects the two team members' work; amend this log if needed.
- [ ] Retain the original Codabench screenshots and add the final EB-NeRD score screenshot when available.
- [ ] Verify all reported metrics by rerunning the relevant command or checking the saved JSON result.
- [ ] Confirm that no raw datasets, model caches, prediction text files, ZIP submissions, or secrets are staged in Git.
