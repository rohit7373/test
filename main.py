# --- DB CONNECTION (Load Dual Config) ---

if not MONGO_URL:
# ... (rest of the block remains the same) ...
        # Load State
        saved = state_collection.find_one({"_id": "global_state"})
        if saved:
            STATE["wallet"] = saved.get("wallet", 1000.0)
            STATE["running"] = saved.get("running", False)
            
            # Load Dual Configs
            if "buy_config" in saved: BUY_CONFIG.update(saved["buy_config"])
            if "sell_config" in saved: SELL_CONFIG.update(saved["sell_config"])
            if "general_config" in saved: GENERAL_CONFIG.update(saved["general_config"])
            
            # ------------------------------------------------------------------
            # START OF REVISED MIGRATION LOGIC (if coming from an older version)
            # ------------------------------------------------------------------
            elif "config" in saved: 
                old_cfg = saved["config"]
                
                # 1. Migrate General Settings
                GENERAL_CONFIG_KEYS = ["qty", "fee", "tp", "sl"]
                for k in GENERAL_CONFIG_KEYS:
                    if k in old_cfg: GENERAL_CONFIG[k] = old_cfg[k]
                
                # 2. Migrate Logic Settings (Applies to both Buy and Sell)
                LOGIC_CONFIG_KEYS = ["stf1", "stf2", "ltf1", "ltf2", "stf_logic", "ltf_logic", "entry_mode"]
                logic_update = {}
                for k in LOGIC_CONFIG_KEYS:
                    if k in old_cfg: logic_update[k] = old_cfg[k]
                    
                BUY_CONFIG.update(logic_update)
                SELL_CONFIG.update(logic_update)
            # ------------------------------------------------------------------
            # END OF REVISED MIGRATION LOGIC
            # ------------------------------------------------------------------
        else:
# ... (rest of the block remains the same) ...
