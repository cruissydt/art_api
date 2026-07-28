
import logging
import zoneinfo
import common

from flask import Flask
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from datetime import datetime

from koha_crm import koha_crm_bp, limiter
from koha_crm_2 import koha_crm_2_bp, limiter


# =========================================================
# 日誌基礎設定 (建議放在 App 初始化之前)
# =========================================================
class TaiwanFormatter(logging.Formatter):
    def converter(self, timestamp):
        # 取得台灣時區
        tz = zoneinfo.ZoneInfo("Asia/Taipei")
        dt = datetime.fromtimestamp(timestamp, tz=tz)
        return dt.timetuple()


# 使用自訂的 Formatter
formatter = TaiwanFormatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

logging.basicConfig(level=logging.INFO, filename="security.log", filemode="a")


# 套用自訂的 Formatter
for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)



# =========================================================
# Flask App 初始化
# =========================================================

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

limiter.init_app(app)


# 註冊 Blueprint
app.register_blueprint(koha_crm_bp, url_prefix="/koha_crm")
app.register_blueprint(koha_crm_2_bp, url_prefix="/koha_crm_2")

CORS(app, origins=common.WHITELIST_ORIGINS)







# =========================================================
# Routes
# =========================================================

@app.route("/")
def home_page():
    return "🌠🌠🌠"



# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    app.run(host=common.FLASK_HOST, port=common.FLASK_PORT, debug=False)
