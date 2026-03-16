# OMEGA PROTOCOL - HARDWARE & NETWORK TOPOLOGY

This document outlines the current hardware and network topology for the Omega Protocol, integrating the latest hybrid network design from the canonical source (`/projects/networks`).

## 1. Hardware Nodes

### Primary Control Node (Current Node)
* **Hostname:** `omega-vm`
* **Location:** Google Cloud VM
* **Specs:** No GPU, 16GB RAM.
* **Role:** Master control node. Primary site for code repository, Gemini CLI, and Git synchronization. Cannot be used for training.

### Central Command Console
* **Hostname:** `zephrymac-studio` (Mac)
* **Specs:** Apple Silicon M4 with 32GB Unified Memory.
* **Role:** The Chief Architect's primary operating machine. Used to access `omega-vm` and serve as the central console for deployment.

### Worker Node: Windows Forge
* **Hostname:** `windows1-w1`
* **Specs:** 
  * Processor: AMD AI Max 395 
  * Memory: 128GB Unified Memory
  * Storage (Internal): 4TB Samsung 990 Pro SSD
  * Storage (External): 8TB USB4 SSD
* **Data Payload:** External SSD contains all Level-2 tick data from 2023 to Jan 2026.
* **Role:** High-performance data forge and deterministic computation.

### Worker Node: Linux Forge
* **Hostname:** `linux1-lx`
* **Specs:** 
  * Processor: AMD AI Max 395
  * Memory: 128GB Unified Memory
  * Storage (Internal): 4TB Samsung 990 Pro SSD
  * Storage (External): 8TB USB4 SSD
* **Data Payload:** External SSD contains all Level-2 tick data from 2023 to Jan 2026.
* **Role:** High-performance data forge and deterministic computation.

---

## 2. Network Topology & SSH Methods

The network operates on a **controlled hybrid design** (no longer a pure Tailscale mesh), utilizing public HK SSH and WireGuard to ensure stable connections bridging the GFW.

### Canonical Authoritative Paths

**1. MacStudio to `omega-vm`**
* **Primary:** `ssh omega-vm-hk` (MacStudio -> HK public SSH -> WireGuard `10.88.0.1`)
* **Secondary:** `ssh omega-vm` (direct Tailscale to `100.122.223.27`)
* **Fallback:** `ssh omega-vm-hk-ts` (MacStudio -> HK Tailscale -> WireGuard `10.88.0.1`)

**2. `omega-vm` to Shenzhen Home Devices**
* **To Linux:** `ssh linux1-lx` (routes via `hk-wg -> localhost:2226`)
* **To MacStudio:** `ssh zephrymac-studio` (routes via `hk-wg -> localhost:2227`)
* **To Windows:** `ssh windows1-w1` (routes via `hk-wg -> localhost:2228`)

**Legacy/Fallback paths (kept active for safety):**
* Jump via Linux: `zephrymac-studio-via-linux1`, `windows1-via-linux1`
* Legacy Mac local forwards: `linux1-back`, `windows1-back` (maintained via `omega-mac-back.service`)

### Key Network Nodes & IPs
| Node | Public IP | Tailscale IP | WireGuard IP | Role |
|---|---|---|---|---|
| **HK** (`vm-0-7-ubuntu`) | `43.161.252.57` | `100.81.234.55` | `10.88.0.2` | Public hub, reverse-SSH landing point, WireGuard peer |
| **US** (`omega-vm`) | `34.27.27.164` | `100.122.223.27` | `10.88.0.1` | Remote controller |
| **Mac Studio** | home LAN | `100.72.87.94` | n/a | Main local workstation |
| **Linux** (`linux1-lx`) | home LAN | `100.64.97.113` | n/a | Always-on jump host |
| **Windows** (`windows1-w1`) | home LAN | `100.123.90.25` | n/a | Home workstation |

### Operational Rules
* **Canonical Truth:** For any discrepancies or further details, defer exclusively to `/projects/networks/README.md` and `/projects/networks/handover.md`. Do not trust older Tailscale-direct or `mac-back`-centric notes under legacy directories.
* **Mac OS Tailscale Changes:** Do not use remote `tailscale up` as a casual change tool; use `tailscale set` to avoid cutting off access paths.