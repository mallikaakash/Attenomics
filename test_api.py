import requests
import json

def test_rest_endpoint():
    print("\nTesting REST API endpoint...")
    url = "http://localhost:8000/compute_scores_batch"
    
    # Sample tweet data
    sample_tweets = {
        "tweets": [
            {
                "text": "ORA is building amazing infrastructure for AI and blockchain integration",
                "likes": 100,
                "likes_day0": 50,
                "likes_day1": 30,
                "likes_day2": 15,
                "likes_day3": 5,
                "likes_total": 100,
                "retweets": 20,
                "retweets_day0": 10,
                "retweets_day1": 6,
                "retweets_day2": 3,
                "retweets_day3": 1,
                "retweets_total": 20,
                "bookmarkCount": 15,
                "bookmarkCount_day0": 8,
                "bookmarkCount_day1": 4,
                "bookmarkCount_day2": 2,
                "bookmarkCount_day3": 1,
                "bookmarkCount_total": 15,
                "views": "1000",
                "views_day0": 500,
                "views_day1": 300,
                "views_day2": 150,
                "views_day3": 50,
                "views_total": "1000",
                "replies": 10,
                "isQuoted": False,
                "isReply": False,
                "isEdited": False,
                "tweetID": "123456789",
                "username": "testuser",
                "name": "Test User",
                "userId": "987654321",
                "timestamp": 1678901234,
                "permanentUrl": "https://twitter.com/testuser/status/123456789",
                "conversationId": "123456789"
            }
        ]
    }
    
    response = requests.post(url, json=sample_tweets)
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))

def test_graphql_endpoint():
    print("\nTesting GraphQL endpoint...")
    url = "http://localhost:8000/graphql"
    
    query = """
    mutation ComputeScoresBatch($tweets: [TweetDataInput!]!) {
        computeTweetScoresBatch(tweets: $tweets) {
            tweetId
            relevancyScore
            attentionScore
            text
        }
    }
    """
    
    variables = {
        "tweets": [{
            "text": "Build with ORA AI & blockchain infrastructure to create decentralized applications with verifiable AI inference on-chain!",
            "likes": 150.0,
            "retweets": 30.0,
            "bookmarkCount": 20.0,
            "views": "5000",
            "tweetId": "9999999999999999999",
            "username": "ora_example",
            "timestamp": 1745090600.0,
            "permanentUrl": "https://twitter.com/ora_example/status/9999999999999999999"
        }]
    }
    
    response = requests.post(url, json={"query": query, "variables": variables})
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_rest_endpoint()
    test_graphql_endpoint() 