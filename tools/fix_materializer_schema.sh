#!/bin/bash
ssh linux1-lx "sed -i 's/\"epiplexity\": np.concatenate(all_epiplexity)/\"epiplexity\": np.concatenate(all_epiplexity), \"symbol\": np.concatenate([[sym] * len(prices) for sym, prices in zip([p[0] for p in pack_list], all_prices)])/' /home/zepher/omega_pure/omega_tensor_materializer.py"
