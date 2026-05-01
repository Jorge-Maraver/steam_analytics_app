# Steam Games Analytics - Cloud Deploy Package

This folder is a self-contained deployment package for the Streamlit app.

## Contents

- `steam_analytics_app/app.py`: Streamlit application.
- `steam_sales_pipeline.py`: original collection/preprocessing helpers required by the applied model workflow.
- `external_applied_model_predict.py`: external Steam/SteamSpy lookup and inference helper.
- `steam_sales_pipeline_output/datasets/`: clean parquet plus the raw post-release CSV needed by the applied model pipeline.
- `steam_sales_pipeline_output/model_results/`: saved models and PCA files used by the app.
- `Dockerfile`: container build for AWS App Runner, ECS/Fargate, EC2, or any Docker host.
- `requirements.txt`: Python dependencies.

## Local Run

```bash
pip install -r requirements.txt
streamlit run steam_analytics_app/app.py
```

For chatbot support:

```bash
set OPENAI_API_KEY=your_key_here
```

On Linux/macOS:

```bash
export OPENAI_API_KEY=your_key_here
```

## Docker Run

```bash
docker build -t steam-analytics-app .
docker run --rm -p 8501:8501 -e OPENAI_API_KEY=your_key_here steam-analytics-app
```

Then open:

```text
http://localhost:8501
```

## AWS Options

### Recommended Simple Option: AWS App Runner

1. Push this folder to a GitHub repository, or build and push the Docker image to Amazon ECR.
2. Create an App Runner service from the repository or ECR image.
3. Set port `8501`.
4. Add environment variables:
   - `OPENAI_API_KEY`
   - optionally `OPENAI_MODEL`
5. Deploy.

### More Flexible Option: ECS/Fargate

1. Build the Docker image.
2. Push it to Amazon ECR.
3. Create an ECS/Fargate service.
4. Expose container port `8501`.
5. Put an Application Load Balancer in front if you need HTTPS/custom domain.
6. Add `OPENAI_API_KEY` as a secret or environment variable.

### Quick Manual Option: EC2

1. Launch an EC2 instance.
2. Install Docker.
3. Copy this folder or pull it from Git.
4. Run the Docker commands above.
5. Open inbound traffic to port `8501`, or place Nginx/HTTPS in front.

## Notes

- The app expects files to remain in the same relative structure included in this package.
- The chatbot works only when `OPENAI_API_KEY` is configured.
- The applied model section calls external Steam and SteamSpy endpoints, so the deployment environment needs outbound internet access.
- Generated external prediction files are ignored by Docker by default and can be treated as temporary runtime output.
