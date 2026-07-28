import logging
import common


from flask import request, jsonify, make_response, Blueprint
from pinecone import Pinecone
from groq import Groq
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

koha_crm_2_bp = Blueprint("koha_crm_2", __name__)


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
    storage_uri=common.REDIS_CONNECTION_STRING,
    strategy="fixed-window",
)


# =========================================================
# 外部服務初始化
# =========================================================
pc = Pinecone(api_key=common.PINECONE_API_KEY).Index(common.PINECONE_INDEX_NAME)
groq_client = Groq(api_key=common.GROQ_API_KEY)


# =========================================================
# Routes
# =========================================================
@koha_crm_2_bp.errorhandler(429)
def ratelimit_handler(e):
    logging.warning(
        f"觸發限流事件 - 來源 IP: {request.remote_addr} - 路徑: {request.path}"
    )
    return make_response(
        jsonify(
            {
                "error": "rate_limit_exceeded",
                "message": "請求太頻繁，請稍後再試",
            }
        ),
        429,
    )


@koha_crm_2_bp.route("/")
def home():
    return "☎️☎️☎️"


@koha_crm_2_bp.route("/api/search", methods=["POST"])
@limiter.limit("5 per minute")
@limiter.limit("100 per hour")
def search_page():
    results = []
    booklist = []

    logging.info(f"收到搜尋請求 - 來源 IP: {request.remote_addr}")

    data = request.get_json(silent=True) or {}

    # 正確範例：{"name": "小明", "age": 25} -> 這是 dict，通過！
    # 錯誤範例："小明" 或 {"小明"} 或 [1, 2, 3] -> 不是 dict，回傳 400 錯誤
    if not isinstance(data, dict):
        return jsonify({"error": "請求格式錯誤"}), 400

    query = data.get("query", "")

    if not isinstance(query, str):
        return jsonify({"error": "請求格式錯誤"}), 400

    query = common.clean_query(query)

    if not query:
        return jsonify({"error": "請求內容空白"}), 400

    if len(query) > 50:
        return jsonify({"error": "請求內容太大"}), 400

    # 使用 LLM 改寫
    rewrite_q = common.rewrite_query(query, groq_client)
    logging.info(f"rewrite['intent']: {rewrite_q['intent']}")
    logging.info(f"rewrite['query']: {rewrite_q['query']}")
    logging.info(f"rewrite['reason']: {rewrite_q['reason']}")

    if rewrite_q["intent"] == "invalid":
        return jsonify({
            "error": "無明確搜尋意圖",
            "message": "請輸入想找的主題，例如：Python、投資、烘焙、心理學..."
        }), 400

    query_embedding = common.ST_MODEL.encode(
        "query: " + rewrite_q["query"]
    ).tolist()


    author_list = common.get_author_list("author_list.txt")
    author_find = common.get_author_find(author_list, query)

    pc_response = pc.query(
        vector=query_embedding,
        top_k=6,
        include_metadata=True,
        filter={"所有名字": {"$in": author_find}} if author_find else None,
    )

    for pc_item in pc_response["matches"]:
        pc_meta = pc_item.metadata or {}

        pc_raw_text = pc_meta.get("轉存向量前的文字", "")
        pc_title = pc_meta.get("題名", "")
        pc_author = ", ".join(pc_meta.get("所有名字", []))
        pc_isbn = ", ".join(pc_meta.get("國際標準書號", []))
        pc_summary = ", ".join(pc_meta.get("AI摘要", []))

        results.append(
            {
                "id": pc_item.id,
                "title": pc_title,
                "authors": pc_author,
                "isbn": pc_isbn,
                "score": round(pc_item.score, 4),
                "summary": pc_summary,
                "llm_msg": "暫無推薦原因",
            }
        )

        temp = "\n".join(
            [
                f"題名：{pc_title}",
                f"作者：{pc_author}",
                f"書籍資料：{pc_raw_text}",
            ]
        )
        booklist.append({"id": pc_item.id, "info": common.get_safe_text(temp)})

    if booklist:
        # llm_msg = common.get_llm_msg(booklist, groq_client)
        llm_msg = {}

        for item in results:
            book_id = item["id"]

            if book_id in llm_msg:
                item["llm_msg"] = llm_msg[book_id]

    logging.info(f"搜尋成功完成 - 回傳 {len(results)} 筆結果")
    return jsonify(
        {
            "query": query,
            "rewrite_query": rewrite_q["query"],
            "results": results,
        }
    )
