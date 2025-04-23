from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
# import strawberry
# from strawberry.fastapi import GraphQLRouter
from typing import Optional, List, Dict, Any

# ─── hyper‑parameters ───────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 512
DEVICE = torch.device("cpu")  # keep everything on CPU by default

# ─── SiameseBert definition ────────────────────────────────────
class SiameseBert(nn.Module):
    def __init__(self):
        super().__init__()
        # this loads real weights onto CPU
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, 256)
        self.relu = nn.ReLU()

    def encode(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output  # [batch, 768]
        return self.relu(self.linear(self.dropout(pooled)))  # [batch, 256]

    def forward(self, ids1, mask1, ids2, mask2):
        return self.encode(ids1, mask1), self.encode(ids2, mask2)

# ─── Model loading function ───────────────────────────────────────
def load_model():
    MODEL_FOLDER = os.path.join(os.path.dirname(__file__), "app/model")
    print("Tokenizer path", os.path.join(MODEL_FOLDER, "TLXD-siamese_tokenizer"))
    print("Model path", os.path.join(MODEL_FOLDER, "TLXD-siamese_bert_contrastive"))
    try:
        tokenizer = BertTokenizer.from_pretrained(os.path.join(MODEL_FOLDER, "TLXD-siamese_tokenizer"), local_files_only=True)
        
        model = SiameseBert()
        state_dict = torch.load(
            os.path.join(MODEL_FOLDER, "TLXD-siamese_bert_contrastive.pth"),
            map_location=DEVICE
        )
        model.load_state_dict(state_dict)
        model.eval()  # keep on CPU; no .to() needed
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

# ─── Anchor text ────────────────────────────────────────────────
FULL_ANCHOR = """
Option 1: Concise Mission Summary
ORA builds chain-agnostic infrastructure bridging AI and blockchain. It enables verifiable AI inference and tokenization of AI models on-chain, empowering developers to build decentralized applications using trustless AI.

Option 2: Extended Mission + Offerings Summary
ORA provides chain-agnostic infrastructure connecting AI and blockchain. Developers can build decentralized applications with verifiable AI using ORA's AI Oracle and tokenized models. Offerings include trustless AI inference, smart contract integration, AI-generated content (NFTs), and sustainable open-source AI funding via Initial Model Offerings (IMOs).

Option 3: Full Token (Comprehensive Description)
ORA builds chain-agnostic infrastructure to bridge AI and blockchain, offering tools for developers to create trustless decentralized applications powered by verifiable AI. Through its AI Oracle and opML-based architecture, ORA enables on-chain inference for large models like LlaMA2 and Stable Diffusion. It supports tokenization of AI models via Initial Model Offerings (IMOs), creating sustainable funding and shared ownership for open-source AI. Use cases include verifiable NFTs, prediction markets, autonomous agents, and risk management in DeFi.

Optional Tokens for Topic-Specific Embeddings:

Token: Verifiable AI Inference
ORA enables verifiable AI inference on-chain using opML and zkML, supporting large models like LlaMA2 and Stable Diffusion. AI outputs are trustless and cryptographically verifiable.

Token: Decentralized AI Oracle
ORA's AI Oracle integrates with smart contracts, offering on-chain AI model access, trustless predictions, and AI-driven logic for DeFi, NFTs, and DAOs.

Token: AI Model Tokenization
Initial Model Offerings (IMOs) allow tokenization of AI models on-chain via ERC-7641, funding open-source AI and distributing revenue to contributors and token holders.
"""

# ─── helper: embed any text ─────────────────────────────────────
@torch.no_grad()
def embed_text(text: str, tokenizer, model):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    ids, mask = enc["input_ids"], enc["attention_mask"]
    emb, _ = model(ids, mask, ids, mask)
    return emb  # [1,256]

# ─── attention‑score calculator ─────────────────────────────────────
def compute_enhanced_attention_score(relevancy, tweet_data):
    # Arbitrary weights in decreasing order for engagement metrics
    w = {
        "likes": 0.15,
        "comments": 0.2, 
        "bookmarks": 0.25,
        "retweets": 0.3,
        "views": 0.1
    }
    
    # Equal weight for all time periods
    time_decay = {
        "day0": 0.25,    
        "day1": 0.25,    
        "day2": 0.25,    
        "day3": 0.25     
    }
    
    # Calculate weighted engagement score with time decay
    engagement_score = 0
    
    # Likes over time
    likes_score = (
        time_decay["day0"] * float(tweet_data.get("likes_day0", 0)) +
        time_decay["day1"] * float(tweet_data.get("likes_day1", 0)) +
        time_decay["day2"] * float(tweet_data.get("likes_day2", 0)) +
        time_decay["day3"] * float(tweet_data.get("likes_day3", 0))
    )
    
    # Retweets over time
    retweets_score = (
        time_decay["day0"] * float(tweet_data.get("retweets_day0", 0)) +
        time_decay["day1"] * float(tweet_data.get("retweets_day1", 0)) +
        time_decay["day2"] * float(tweet_data.get("retweets_day2", 0)) +
        time_decay["day3"] * float(tweet_data.get("retweets_day3", 0))
    )
    
    # Bookmarks over time
    bookmarks_score = (
        time_decay["day0"] * float(tweet_data.get("bookmarkCount_day0", 0)) +
        time_decay["day1"] * float(tweet_data.get("bookmarkCount_day1", 0)) +
        time_decay["day2"] * float(tweet_data.get("bookmarkCount_day2", 0)) +
        time_decay["day3"] * float(tweet_data.get("bookmarkCount_day3", 0))
    )
    
    # Views over time
    views_score = (
        time_decay["day0"] * float(tweet_data.get("views_day0", 0)) +
        time_decay["day1"] * float(tweet_data.get("views_day1", 0)) +
        time_decay["day2"] * float(tweet_data.get("views_day2", 0)) +
        time_decay["day3"] * float(tweet_data.get("views_day3", 0))
    )
    
    # Calculate total engagement score
    engagement_score = (
        w["likes"] * likes_score +
        w["retweets"] * retweets_score +
        w["bookmarks"] * bookmarks_score +
        w["comments"] * float(tweet_data.get("replies", 0)) +
        w["views"] * views_score
    )
    
    # Ignore
    if tweet_data.get("isQuoted", False):
        engagement_score *= 1  
    
    # Scale the score properly
    engagement_score = engagement_score / 1000  # Normalize to reasonable range
    
    # Compute the final attention score: relevancy * engagement
    return relevancy * (1 + engagement_score)  # Ensure relevancy is always a factor

# Initialize FastAPI
app = FastAPI(title="ORA Tweet Similarity API")

# Load model on startup
model = None
tokenizer = None
anchor_emb = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, anchor_emb
    try:
        model, tokenizer = load_model()
        anchor_emb = embed_text(FULL_ANCHOR, tokenizer, model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")

# ─── REST API Endpoints ────────────────────────────────────────
class TweetDataModel(BaseModel):
    text: str
    likes: float = 0
    likes_day0: float = 0
    likes_day1: float = 0
    likes_day2: float = 0
    likes_day3: float = 0
    likes_total: float = 0
    retweets: float = 0
    retweets_day0: float = 0
    retweets_day1: float = 0
    retweets_day2: float = 0
    retweets_day3: float = 0
    retweets_total: float = 0
    bookmarkCount: float = 0
    bookmarkCount_day0: float = 0
    bookmarkCount_day1: float = 0
    bookmarkCount_day2: float = 0
    bookmarkCount_day3: float = 0
    bookmarkCount_total: float = 0
    views: str = "0"
    views_day0: float = 0
    views_day1: float = 0
    views_day2: float = 0
    views_day3: float = 0
    views_total: str = "0"
    replies: float = 0
    isQuoted: bool = False
    isReply: bool = False
    isEdited: bool = False
    tweetID: str = ""
    username: str = ""
    name: str = ""
    userId: str = ""
    timestamp: float = 0
    permanentUrl: str = ""
    conversationId: str = ""

class TweetArrayRequest(BaseModel):
    tweets: List[TweetDataModel]

class TweetScoreResponse(BaseModel):
    tweetID: str
    relevancy_score: float
    attention_score: float
    text: str

class TweetArrayResponse(BaseModel):
    scores: List[TweetScoreResponse]

@app.post("/compute_scores_batch", response_model=TweetArrayResponse)
async def compute_scores_batch(request: TweetArrayRequest):
    global model, tokenizer, anchor_emb
    
    if model is None or tokenizer is None or anchor_emb is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    scores = []
    
    for tweet in request.tweets:
        # Embed tweet & compute cosine similarity
        tweet_emb = embed_text(tweet.text, tokenizer, model)
        sim_score = nn.functional.cosine_similarity(tweet_emb, anchor_emb, dim=1).item()
        
        # Compute enhanced attention score
        attention = compute_enhanced_attention_score(sim_score, tweet.dict())
        
        scores.append(TweetScoreResponse(
            tweetID=tweet.tweetID,
            relevancy_score=sim_score,
            attention_score=attention,
            text=tweet.text
        ))
    
    return TweetArrayResponse(scores=scores)

# ─── GraphQL Schema ────────────────────────────────────────────
# @strawberry.type
# class TweetScore:
#     tweet_id: str
#     relevancy_score: float
#     attention_score: float
#     text: str

# @strawberry.input
# class TweetDataInput:
#     text: str
#     likes: float = 0
#     likes_day0: float = 0
#     likes_day1: float = 0
#     likes_day2: float = 0
#     likes_day3: float = 0
#     likes_total: float = 0
#     retweets: float = 0
#     retweets_day0: float = 0
#     retweets_day1: float = 0
#     retweets_day2: float = 0
#     retweets_day3: float = 0
#     retweets_total: float = 0
#     bookmark_count: float = 0
#     bookmark_count_day0: float = 0
#     bookmark_count_day1: float = 0
#     bookmark_count_day2: float = 0
#     bookmark_count_day3: float = 0
#     bookmark_count_total: float = 0
#     views: str = "0"
#     views_day0: float = 0
#     views_day1: float = 0
#     views_day2: float = 0
#     views_day3: float = 0
#     views_total: str = "0"
#     replies: float = 0
#     is_quoted: bool = False
#     is_reply: bool = False
#     is_edited: bool = False
#     tweet_id: str = ""
#     username: str = ""
#     name: str = ""
#     user_id: str = ""
#     timestamp: float = 0
#     permanent_url: str = ""
#     conversation_id: str = ""

# @strawberry.type
# class Query:
#     @strawberry.field
#     def hello(self) -> str:
#         return "Hello World!"

# @strawberry.type
# class Mutation:
#     @strawberry.mutation
#     def compute_tweet_scores_batch(self, tweets: List[TweetDataInput]) -> List[TweetScore]:
#         global model, tokenizer, anchor_emb
        
#         if model is None or tokenizer is None or anchor_emb is None:
#             raise Exception("Model not loaded")
        
#         scores = []
        
#         for tweet in tweets:
#             # Convert GraphQL input to REST API format
#             tweet_data = {
#                 "text": tweet.text,
#                 "likes": tweet.likes,
#                 "likes_day0": tweet.likes_day0,
#                 "likes_day1": tweet.likes_day1,
#                 "likes_day2": tweet.likes_day2,
#                 "likes_day3": tweet.likes_day3,
#                 "likes_total": tweet.likes_total,
#                 "retweets": tweet.retweets,
#                 "retweets_day0": tweet.retweets_day0,
#                 "retweets_day1": tweet.retweets_day1,
#                 "retweets_day2": tweet.retweets_day2,
#                 "retweets_day3": tweet.retweets_day3,
#                 "retweets_total": tweet.retweets_total,
#                 "bookmarkCount": tweet.bookmark_count,
#                 "bookmarkCount_day0": tweet.bookmark_count_day0,
#                 "bookmarkCount_day1": tweet.bookmark_count_day1,
#                 "bookmarkCount_day2": tweet.bookmark_count_day2,
#                 "bookmarkCount_day3": tweet.bookmark_count_day3,
#                 "bookmarkCount_total": tweet.bookmark_count_total,
#                 "views": tweet.views,
#                 "views_day0": tweet.views_day0,
#                 "views_day1": tweet.views_day1,
#                 "views_day2": tweet.views_day2,
#                 "views_day3": tweet.views_day3,
#                 "views_total": tweet.views_total,
#                 "replies": tweet.replies,
#                 "isQuoted": tweet.is_quoted,
#                 "isReply": tweet.is_reply,
#                 "isEdited": tweet.is_edited,
#                 "tweetID": tweet.tweet_id,
#                 "username": tweet.username,
#                 "name": tweet.name,
#                 "userId": tweet.user_id,
#                 "timestamp": tweet.timestamp,
#                 "permanentUrl": tweet.permanent_url,
#                 "conversationId": tweet.conversation_id
#             }
            
#             # Embed tweet & compute cosine similarity
#             tweet_emb = embed_text(tweet.text, tokenizer, model)
#             sim_score = nn.functional.cosine_similarity(tweet_emb, anchor_emb, dim=1).item()
            
#             # Compute enhanced attention score
#             attention = compute_enhanced_attention_score(sim_score, tweet_data)
            
#             scores.append(TweetScore(
#                 tweet_id=tweet.tweet_id,
#                 relevancy_score=sim_score,
#                 attention_score=attention,
#                 text=tweet.text
#             ))
        
#         return scores

# schema = strawberry.Schema(query=Query, mutation=Mutation)
# graphql_app = GraphQLRouter(schema)

# app.include_router(graphql_app, prefix="/graphql")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)