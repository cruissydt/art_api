import logging
import common


from flask import request, jsonify, make_response, Blueprint
from pinecone import Pinecone
from groq import Groq

# 是否過濾使用者指令並改寫指令
is_rewrite = 0
# 是否新增推薦原因
is_llm_msg = 0

koha_crm_bp = Blueprint("koha_crm", __name__)


# =========================================================
# 外部服務初始化
# =========================================================
pc = Pinecone(api_key=common.PINECONE_API_KEY).Index(common.PINECONE_INDEX_NAME)
groq_client = Groq(api_key=common.GROQ_API_KEY)


# =========================================================
# Routes
# =========================================================
@koha_crm_bp.errorhandler(429)
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


@koha_crm_bp.errorhandler(500)
def internal_error_handler(e):
    logging.exception(
        f"未預期的內部錯誤 - 來源 IP: {request.remote_addr} - 路徑: {request.path}"
    )
    return make_response(
        jsonify(
            {
                "error": "internal_error",
                "message": "系統忙碌中，請稍後再試",
            }
        ),
        500,
    )


@koha_crm_bp.route("/")
def home():
    return "☎️☎️☎️"


@koha_crm_bp.route("/api/search", methods=["POST"])
@common.limiter.limit("5 per minute")
@common.limiter.limit("100 per hour")
def search_page():
    results = []

    logging.info(f"收到搜尋請求 - 來源 IP: {request.remote_addr}")

    data = request.get_json(silent=True) or {}

    # 正確範例：{"name": "小明", "age": 25} -> 這是 dict，通過！
    # 錯誤範例："小明" 或 {"小明"} 或 [1, 2, 3] -> 不是 dict，回傳 400 錯誤
    if not isinstance(data, dict):
        return (
            jsonify({"error": "invalid_format", "message": "請求格式錯誤"}),
            400,
        )

    query = data.get("query", "")

    if not isinstance(query, str):
        return (
            jsonify({"error": "invalid_format", "message": "請求格式錯誤"}),
            400,
        )

    query = common.clean_query(query)

    if not query:
        return (
            jsonify({"error": "empty_query", "message": "請求內容空白"}),
            400,
        )

    if len(query) > 50:
        return (
            jsonify({"error": "query_too_large", "message": "請求內容太大"}),
            413,
        )

    if is_rewrite:
        # 不建議使用 LLM 改寫，因為額度有限和使用者輸入被檔住，出現會很煩的"錯誤:非找書意圖指令"訊息
        rewrite_q = common.rewrite_query(query, groq_client)
        logging.info(f"rewrite['valid']: {rewrite_q['valid']}")
        logging.info(f"rewrite['query']: {rewrite_q['query']}")
        logging.info(f"rewrite['reason']: {rewrite_q['reason']}")

        if rewrite_q["valid"] == False:
            return (
                jsonify({"error": "unclear_intent", "message": rewrite_q["reason"]}),
                400,
            )
        ising_rewrite = rewrite_q["ising_rewrite"]
        """
        query_embedding = common.ST_MODEL.encode(
            "query: " + rewrite_q["query"]
        ).tolist()
        """
        query_embedding = common.ST_MODEL.encode(rewrite_q["query"]).tolist()
    else:
        ising_rewrite = 0
        # query_embedding = common.ST_MODEL.encode("query: " + query).tolist()
        query_embedding = common.ST_MODEL.encode(query).tolist()

    temp = common.get_author_list("koha_crm_author_list.txt")
    author_find = common.get_author_find(temp, query)

    pc_response = pc.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True,
        filter={"所有名字": {"$in": author_find}} if author_find else None,
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

    ising_llm_msg = 0
    if is_llm_msg:
        try:
            llm_msg = common.get_llm_msg(results, groq_client)
        except Exception:
            logging.exception("get_llm_msg 呼叫失敗")
            llm_msg = {}

        for item in results:
            book_id = item["id"]
            if book_id in llm_msg:
                item["llm_msg"] = llm_msg[book_id]
                ising_llm_msg = 1

    # logging.info(f"搜尋成功完成 - 回傳 {len(results)} 筆結果")
    return jsonify(
        {
            "results": results,
            "query": query,
            "ising_rewrite": ising_rewrite,
            "ising_llm_msg": ising_llm_msg,
        }
    )
