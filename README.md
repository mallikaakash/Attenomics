# Setting Up

### 1 - Clone the repo 
```
git clone https://github.com/mallikaakash/Attenomics
```

### 2 - cd into the folder
```
cd attenomics-ai-server
```

### 3 - Install required ackages
```
pip install -r requirements.txt
```

### 4 - run the fastapi server
```
uvicorn main:app --reload
```

### 5 - test it by sending a payloa to the enpoint REST API endpoint (detailed below)

# Tweet Analysis Service

A machine learning service that analyzes tweets using a BERT-based Siamese neural network to compute relevancy and attention scores.

## System Architecture

The service uses a BERT-based Siamese neural network model for tweet analysis. The model files are not included in this repository due to their large size. You will need to download them separately.

### Model Files

The model files are required for the service to function:
- Model path: `app/model/TLXD-siamese_bert_contrastive`
- Tokenizer path: `app/model/TLXD-siamese_tokenizer`

To download the model files:
1. Create the model directory: `mkdir -p app/model`
2. Download the model files from [Google Drive](https://drive.google.com/drive/folders/your-folder-id) (link to be provided)
3. Extract the files to the `app/model` directory

## API Endpoints

The service provides two endpoints for tweet analysis:

### 1. REST API Endpoint

```
POST http://localhost:8000/compute_scores_batch
```

#### Request Format
```json
{
    "tweets": [{
        "text": "Tweet text content",
        "likes": float,
        "retweets": float,
        "bookmarkCount": float,
        "views": string,
        "tweetID": string,
        "username": string,
        "timestamp": float,
        "permanentUrl": string
    }]
}
```

#### Response Format
```json
{
    "scores": [{
        "tweetID": string,
        "relevancy_score": float,
        "attention_score": float,
        "text": string
    }]
}
```

### 2. GraphQL Endpoint

```
POST http://localhost:8000/graphql
```

#### Query Format
```graphql
mutation ComputeScoresBatch($tweets: [TweetDataInput!]!) {
    computeTweetScoresBatch(tweets: $tweets) {
        tweetId
        relevancyScore
        attentionScore
        text
    }
}
```

#### Variables Format
```json
{
    "tweets": [{
        "text": "Tweet text content",
        "likes": float,
        "retweets": float,
        "bookmarkCount": float,
        "views": string,
        "tweetId": string,
        "username": string,
        "timestamp": float,
        "permanentUrl": string
    }]
}
```

## Batch Processing

Both endpoints support batch processing of tweets. You can send multiple tweets in a single request, and the service will return scores for all tweets in the batch.

## Score Calculation

The service calculates two types of scores:

1. **Relevancy Score** (0-1): Measures how relevant the tweet content is to a specific domain or topic.
2. **Attention Score** (0-1): Measures the potential engagement or attention the tweet might receive.

The scores are calculated using a BERT-based Siamese neural network that has been trained to understand tweet content and engagement patterns.

## Technical Requirements

- Python 3.8+
- TensorFlow 2.x with CUDA support
- FastAPI
- GraphQL (Strawberry)
- CUDA-compatible GPU (recommended)

## Environment Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export TF_ENABLE_ONEDNN_OPTS=0  # Disable oneDNN custom operations
export TF_CPP_MIN_LOG_LEVEL=2   # Reduce TensorFlow logging
```

3. Ensure CUDA is properly configured:
```bash
nvidia-smi  # Verify GPU is detected
```

4. Download and set up model files:
```bash
mkdir -p app/model
# Download model files from Google Drive
# Extract files to app/model directory
```

## Troubleshooting

### Common Issues

1. **CUDA Factory Registration Warnings**
   - These warnings are normal and can be safely ignored
   - They occur because TensorFlow tries to register CUDA factories multiple times
   - No impact on functionality

2. **Model Reloading**
   - The service uses FastAPI's auto-reload feature
   - To disable auto-reload, start the server with:
   ```bash
   uvicorn app.main:app --reload=false
   ```

3. **GPU Memory Issues**
   - If you encounter GPU memory errors, try:
   ```python
   import tensorflow as tf
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
   ```

4. **Performance Optimization**
   - Enable AVX2, AVX512F, AVX512_VNNI, and FMA instructions for better CPU performance
   - Set environment variables:
   ```bash
   export TF_ENABLE_ONEDNN_OPTS=0
   export TF_CPP_MIN_LOG_LEVEL=2
   ```

## Example Usage

```python
import requests

# REST API Example
response = requests.post(
    "http://localhost:8000/compute_scores_batch",
    json={
        "tweets": [{
            "text": "Example tweet text",
            "likes": 150.0,
            "retweets": 30.0,
            "bookmarkCount": 20.0,
            "views": "5000",
            "tweetID": "123456789",
            "username": "example_user",
            "timestamp": 1745090600.0,
            "permanentUrl": "https://twitter.com/example_user/status/123456789"
        }]
    }
)

# GraphQL Example
response = requests.post(
    "http://localhost:8000/graphql",
    json={
        "query": """
        mutation ComputeScoresBatch($tweets: [TweetDataInput!]!) {
            computeTweetScoresBatch(tweets: $tweets) {
                tweetId
                relevancyScore
                attentionScore
                text
            }
        }
        """,
        "variables": {
            "tweets": [{
                "text": "Example tweet text",
                "likes": 150.0,
                "retweets": 30.0,
                "bookmarkCount": 20.0,
                "views": "5000",
                "tweetId": "123456789",
                "username": "example_user",
                "timestamp": 1745090600.0,
                "permanentUrl": "https://twitter.com/example_user/status/123456789"
            }]
        }
    }
)
```

## Development

1. **Running Tests**
```bash
python test_api.py
```

2. **Starting the Server**
```bash
# Development mode (with auto-reload)
uvicorn app.main:app --reload

# Production mode (without auto-reload)
uvicorn app.main:app --reload=false
```

3. **Monitoring**
- Check server logs for any warnings or errors
- Monitor GPU memory usage with `nvidia-smi`
- Use the test script to verify API functionality
