# Agent Flow

This diagram shows the full runtime path from the initial user question through
tool use, quality review, summarization, and optional follow-up research.

```mermaid
flowchart TD
  U["User input<br/>Agent: CLI entrypoint<br/>Target: initial research question<br/>Tools: argparse, dotenv<br/>Output: first user message"]
  Runtime["Runtime state<br/>Agent: run_research loop<br/>Does: tracks messages, usage, tool history, quality flags<br/>Target: current review question<br/>Output: message list to ResearchAgent"]

  U --> Runtime

  subgraph ResearchLoop["Small loop 1: research/tool loop"]
    Research["ResearchAgent<br/>Does: searches, reads, asks clarifying questions, drafts cited answer<br/>Target: current research question<br/>Tools: brave_search, fetch_page, query_user<br/>Output: tool_use request or candidate final answer"]
    ToolRunner["Python tool runner<br/>Does: validates args and executes requested tool<br/>Target: model tool_use block<br/>Tools: Brave Search API, httpx page fetch, terminal input<br/>Output: tool_result observation"]

    Research -->|"tool_use"| ToolRunner
    ToolRunner -->|"observation appended to messages"| Research
  end

  Runtime --> Research

  subgraph QualityLoop["Small loop 2: quality audit/retry loop"]
    Quality["QualityCheckAgent<br/>Does: audits correctness, directness, and comprehensiveness<br/>Target: candidate answer plus question and tool history<br/>Tools: brave_search, fetch_page, query_user<br/>Output: JSON metrics, flags, rejection"]
    AuditTools["Python tool runner<br/>Does: runs audit-time verification tools<br/>Target: quality tool_use block<br/>Tools: same tool set as research<br/>Output: audit observation"]
    SchemaRetry["Quality schema retry<br/>Agent: QualityCheckAgent<br/>Does: repairs unparsable quality output<br/>Target: required JSON contract<br/>Output: corrected quality JSON"]

    Quality -->|"audit tool_use"| AuditTools
    AuditTools -->|"audit observation appended"| Quality
    Quality -->|"invalid JSON"| SchemaRetry
    SchemaRetry -->|"retry prompt"| Quality
  end

  Research -->|"candidate final answer"| Quality
  Quality -->|"flags fail: rejection appended to messages"| Research

  Summary["SummaryAgent<br/>Does: writes only the markdown TL;DR<br/>Target: approved detailed answer<br/>Tools: none<br/>Output: TL;DR"]
  Print["CLI output<br/>Agent: runtime printer<br/>Does: prints TL;DR, detailed report, sources, usage<br/>Target: accepted answer<br/>Tools: terminal output<br/>Output: report shown to user"]
  FollowUp{"Follow-up entered?"}
  Builder["QuestionBuilderAgent<br/>Does: rewrites follow-up into standalone next question with comments<br/>Target: previous question, approved answer, new user prompt<br/>Tools: none<br/>Output: next research target"]
  Done["Finish<br/>Agent: CLI<br/>Does: prints done summary and exits<br/>Output: process complete"]

  Quality -->|"all flags pass"| Summary
  Summary --> Print
  Print --> FollowUp
  FollowUp -->|"no: user presses Enter"| Done
  FollowUp -->|"yes: user types follow-up"| Builder
  Builder -->|"append rewritten question to messages"| Runtime
```

## Agent Targets And Outputs

| Step | Agent | Target | Tools | Output destination |
| --- | --- | --- | --- | --- |
| Input setup | CLI entrypoint and `run_research` | User's starting question | `argparse`, `.env` loading | Initial `messages` list |
| Research/tool loop | `ResearchAgent` | Current research question | `brave_search`, `fetch_page`, `query_user` | Tool calls go to `run_tool`; candidate answers go to quality review |
| Tool execution | Python tool runner | Model `tool_use` arguments | Brave Search API, `httpx`, terminal input | `tool_result` observations appended to the message list |
| Quality audit loop | `QualityCheckAgent` | Candidate answer, target question, and tool history | Same three tools when verification helps | Structured quality JSON |
| Rejection path | `run_research` | Failed quality flags and rejection text | Message append | Rejection returns to `ResearchAgent` for another draft |
| Acceptance path | `SummaryAgent` | Approved detailed answer | None | Markdown TL;DR |
| User-visible output | CLI printer | Accepted answer and summary | Terminal output | TL;DR, final cited report, usage summary |
| Follow-up path | `QuestionBuilderAgent` | Prior question, approved answer, and new prompt | None | New standalone research question appended to `messages` |
