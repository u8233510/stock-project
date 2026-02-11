def rebuild_latest_branch_weighted_cost(conn, universe, t_start, t_end, log, failed_log, sync_func):
    """Run rolling weighted-cost rebuild after ingest is completed."""
    log("🧮 開始更新 branch_weighted_cost（僅最新 5/20/60 rolling 快照）...")
    for stock in universe:
        sid = stock["stock_id"]
        try:
            updated = sync_func(conn, sid, t_start, t_end)
            if updated:
                log(f"    ✅ [{sid}] weighted_cost 更新基準日: {updated[0]}")
        except Exception as e:
            log(f"    ❌ [{sid}] weighted_cost 更新失敗: {e}")
            failed_log.append(f"{sid} branch_weighted_cost: {e}")


def format_snapshot_caption(row):
    return f"淨張數: {int(row[1])} | 集中度: {float(row[2]):.2f}% | 截止日: {row[3]}"
