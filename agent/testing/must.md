### 1. Test-Driven Development (TDD) MandateCoverage Requirement:
     - For every functional modification or system-wide change, the agent must verify the existence of a corresponding test case. all the test should be defined in and retrived from ./tests/ folder, narrow tests should be grouped together and be added to a new folder. for example : 

     tests/
        dataValiation/
        botResponse/
        Latency/
        integration/
        unit/
        .
        .
        .
        etc



     - Gap Remediation: If a relevant test does not exist for the scope of the change, the agent is strictly required to implement one before proceeding with the deployment of the code. 

     ALL TEST MUST BE DETERMINISTIC
     
### 2. Regression & Backward CompatibilityPost-Change Validation:
    - After any significant architectural or logic update, a full suite of regression tests must be executed.
    - Compatibility Check: The agent must explicitly confirm that new changes do not break existing downstream dependencies or legacy interfaces. 
### 3. Latency & Scalability BenchmarkingPerformance Baselines:
     - For every update, the agent must measure system response speed and assess changes in the scalability profile.
    -  Efficiency Verification:If a performance test already exists: Execute the test and log the delta in measurements (e.g., latency $+/-$ ms).  If no test exists: Author a new benchmarking test to capture these metrics. 
### 4. Metric-Driven Feedback LoopsComprehensive Logging:
    -  All system behaviors, errors, and performance deltas must be captured in persistent logs.
    -  Degradation Thresholds:
        - The agent must compare the current run's metrics against the most recent historical baseline. 
        -  Automatic Rollback/Correction: If the metrics indicate a performance "downgrade" (e.g., increased latency or reduced throughput), the agent must immediately revert the changes or apply optimizations until the metrics meet or exceed the previous baseline.  