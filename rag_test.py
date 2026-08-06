import logging
import common


from flask import request, jsonify, make_response, Blueprint
from pinecone import Pinecone
from groq import Groq

rag_test_bp = Blueprint("rag_test", __name__)


# =========================================================
# 外部服務初始化
# =========================================================
pc = Pinecone(api_key=common.PINECONE_API_KEY).Index(common.PINECONE_INDEX_NAME)
groq_client = Groq(api_key=common.GROQ_API_KEY)


# =========================================================
# Routes
# =========================================================

@rag_test_bp.route("/")
def home():
    return "🦒🦒🦒"


@rag_test_bp.route("/api/search", methods=["POST"])
def search_page():
    results = []

    data = request.get_json(silent=True) or {}

    query = data.get("query", "")

    query = common.clean_query(query)

    query_embedding = common.ST_MODEL.encode("query: " + query).tolist()

    pc_response = pc.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True,
    )

    for pc_item in pc_response["matches"]:
        pc_meta = pc_item.metadata or {}

        pc_raw_text = pc_meta.get("轉存向量前的文字", "")
        pc_title = pc_meta.get("題名", "")
        pc_author = ", ".join(pc_meta.get("所有名字", []))
        pc_isbn = ", ".join(pc_meta.get("國際標準書號", []))
        pc_summary = ", ".join(pc_meta.get("RAG摘要", []))

        results.append(
            {
                "id": pc_item.id,
                "score": round(pc_item.score, 4),
                "raw_text": pc_raw_text,
                "title": pc_title,
                "authors": pc_author,
                "isbn": pc_isbn,
                "summary": pc_summary,
                "llm_msg": "暫無推薦原因",
            }
        )

    return jsonify(
        {
            "results": results,
            "query": query,
        }
    )
