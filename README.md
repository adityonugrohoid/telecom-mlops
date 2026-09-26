# telecom-mlops

Six telecom machine-learning use cases (churn, root cause, anomaly
detection, QoE, capacity forecasting and network optimization) running a
daily drift loop on one shared MLOps pipeline. Each use case owns its own
synthetic data generator and a simulated drift calendar; the pipeline
validates each day's data, detects drift, retrains, and promotes a new
model only when it beats the live one. The data and environments are
simulated; the automation is real.

## Quickstart

```bash
git clone https://github.com/adityonugrohoid/telecom-mlops.git
cd telecom-mlops
```

The repo is being built. The pipeline and the first use case land in the
next pull requests, and this section will then carry the commands that
run them.

## License

MIT, see [LICENSE](LICENSE).

## Author

Adityo Nugroho ([adityonugroho.com](https://adityonugroho.com)),
building with a Claude Code agentic workflow.
