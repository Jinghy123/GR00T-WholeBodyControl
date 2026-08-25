#!/usr/bin/env bash
# Enforce the Dexmate side of the production G1-wired network contract.
#
# This script never writes /etc/resolv.conf, never configures a DNS server, and
# never changes the system default route. `prepare` edits only the dedicated
# NetworkManager profile so it CANNOT contribute DNS/default routes. `activate`
# lets NetworkManager install that profile's direct connected route only;
# `check` is completely read-only.

set -Eeuo pipefail

PROFILE="g1 TELEOP"
IFACE="enp4s0"
LOCAL_CIDR="192.168.123.222/16"
LOCAL_IP="192.168.123.222"
ROBOT_IP="192.168.123.164"
WM_IP="192.168.123.240"
AUTOCONNECT_PRIORITY="100"

die() {
    echo "[g1-network] ERROR: $*" >&2
    exit 1
}

ok() {
    echo "[g1-network] OK: $*"
}

usage() {
    cat <<'EOF'
Usage: g1_teleop_network.sh {check [--no-wm]|activate|prepare|status}

  check     Read-only, fail-closed production contract check.
            --no-wm drops the 192.168.123.240 route leg, for GT-goal runs that
            read goals off local disk and never contact the WM machine.
  activate  Safely activate the already-prepared g1 TELEOP profile, then check.
  prepare   One-time hardening of ONLY the g1 TELEOP profile; does not activate it.
  status    Read-only diagnostic summary; never fails because of profile drift.

Production contract:
  profile/interface  g1 TELEOP / enp4s0
  Dexmate address    192.168.123.222/16
  robot / WM         192.168.123.164 / 192.168.123.240, direct (no gateway)
  wired DNS/default  none (IPv4 and IPv6)

The script never touches the Wi-Fi profile, system DNS, /etc/resolv.conf, or
the system default route. `activate` refuses to switch profiles if the current SSH
management path uses enp4s0.
EOF
}

need_commands() {
    local cmd
    for cmd in nmcli ip resolvectl; do
        command -v "$cmd" >/dev/null 2>&1 || die "required command not found: $cmd"
    done
}

nm_prop() {
    local field="$1" value
    value="$(nmcli -g "$field" connection show "$PROFILE" 2>/dev/null)" \
        || die "NetworkManager profile '$PROFILE' is missing or unreadable"
    [[ "$value" != "--" ]] || value=""
    printf '%s' "$value"
}

expect_prop() {
    local field="$1" expected="$2" actual
    actual="$(nm_prop "$field")"
    [[ "$actual" == "$expected" ]] || die \
        "profile '$PROFILE' has $field='${actual:-<empty>}', expected '${expected:-<empty>}'; run '$0 prepare'"
}

check_profile() {
    nmcli connection show "$PROFILE" >/dev/null 2>&1 \
        || die "NetworkManager profile '$PROFILE' does not exist"
    expect_prop connection.interface-name "$IFACE"
    expect_prop connection.autoconnect "yes"
    expect_prop connection.autoconnect-priority "$AUTOCONNECT_PRIORITY"
    expect_prop ipv4.method "manual"
    expect_prop ipv4.addresses "$LOCAL_CIDR"
    expect_prop ipv4.gateway ""
    expect_prop ipv4.dns ""
    expect_prop ipv4.ignore-auto-dns "yes"
    expect_prop ipv4.never-default "yes"
    expect_prop ipv6.method "disabled"
    expect_prop ipv6.gateway ""
    expect_prop ipv6.dns ""
    expect_prop ipv6.ignore-auto-dns "yes"
    expect_prop ipv6.never-default "yes"
    ok "profile '$PROFILE' is pinned and cannot supply DNS/default routes"
}

route_line() {
    ip -4 route get "$1" 2>/dev/null | head -n 1
}

check_direct_route() {
    local target="$1" label="$2" route
    route="$(route_line "$target")" || true
    [[ -n "$route" ]] || die "no route to $label $target"
    case " $route " in
        *" via "*) die "$label route is indirect: $route" ;;
    esac
    case " $route " in
        *" dev $IFACE "*) ;;
        *) die "$label route does not use $IFACE: $route" ;;
    esac
    case " $route " in
        *" src $LOCAL_IP "*) ;;
        *) die "$label route source is not $LOCAL_IP: $route" ;;
    esac
    ok "$label route is direct: dev $IFACE src $LOCAL_IP"
}

link_dns_values() {
    local line values
    line="$(resolvectl dns "$IFACE" 2>/dev/null)" \
        || die "cannot inspect effective DNS for $IFACE"
    values="${line#*:}"
    values="${values#${values%%[![:space:]]*}}"
    values="${values%${values##*[![:space:]]}}"
    printf '%s' "$values"
}

check_effective_link_isolation() {
    local active cidr dns_values default_v4 default_v6 resolver_default
    active="$(nmcli -g GENERAL.CONNECTION device show "$IFACE" 2>/dev/null)" \
        || die "cannot inspect NetworkManager device $IFACE"
    [[ "$active" == "$PROFILE" ]] \
        || die "$IFACE is using '${active:-<none>}', not '$PROFILE'; run '$0 activate'"

    cidr="$(ip -o -4 address show dev "$IFACE" scope global 2>/dev/null \
        | awk '{print $4}')"
    [[ "$cidr" == "$LOCAL_CIDR" ]] \
        || die "$IFACE address is '${cidr:-<none>}', expected $LOCAL_CIDR"

    default_v4="$(ip -4 route show default dev "$IFACE" 2>/dev/null)"
    default_v6="$(ip -6 route show default dev "$IFACE" 2>/dev/null)"
    [[ -z "$default_v4" && -z "$default_v6" ]] \
        || die "$IFACE contributes a default route (forbidden)"

    dns_values="$(link_dns_values)"
    [[ -z "$dns_values" || "$dns_values" == "none" ]] \
        || die "$IFACE contributes effective DNS servers: $dns_values"
    resolver_default="$(resolvectl default-route "$IFACE" 2>/dev/null)" \
        || die "cannot inspect resolver default-route state for $IFACE"
    case "$resolver_default" in
        *": no") ;;
        *) die "$IFACE is a resolver default route: $resolver_default" ;;
    esac
    ok "$IFACE has $LOCAL_CIDR and contributes no IPv4/IPv6 default route or DNS"
}

# --no-wm drops the WM leg. The GT-goal client reads its goals off local disk and
# never opens a socket to 192.168.123.240, so demanding a route to a machine that
# need not even be powered would block an otherwise single-machine run. The robot
# leg stays mandatory either way: camera and neck still come over the wire.
check_contract() {
    local want_wm=1
    case "${1:-}" in
        --no-wm) want_wm=0 ;;
        "") ;;
        *) die "check: unknown option ${1}" ;;
    esac
    need_commands
    check_profile
    check_effective_link_isolation
    check_direct_route "$ROBOT_IP" "G1"
    if ((want_wm)); then
        check_direct_route "$WM_IP" "WM"
        ok "production G1-wired network contract passed"
    else
        ok "G1-wired network contract passed (WM leg skipped: --no-wm)"
    fi
}

management_guard() {
    local peer route ssh_connection ssh_client
    ssh_connection="${SSH_CONNECTION:-}"
    ssh_client="${SSH_CLIENT:-}"
    peer="${ssh_connection%% *}"
    [[ -n "$peer" ]] || peer="${ssh_client%% *}"
    if [[ -z "$peer" ]]; then
        echo "[g1-network] no SSH peer detected; continuing with interface-local activation"
        return
    fi
    route="$(ip route get "$peer" 2>/dev/null | head -n 1)" || \
        die "cannot determine management route to SSH peer $peer"
    case " $route " in
        *" dev $IFACE "*)
            die "SSH management path to $peer uses $IFACE; refusing remote profile switch (use a local console or another management link)"
            ;;
    esac
    ok "management path to $peer does not use $IFACE"
}

prepare_profile() {
    need_commands
    nmcli connection show "$PROFILE" >/dev/null 2>&1 \
        || die "NetworkManager profile '$PROFILE' does not exist; create it from a local console first"
    echo "[g1-network] hardening ONLY '$PROFILE'; Wi-Fi, system DNS and live routes are untouched"
    nmcli connection modify "$PROFILE" \
        connection.interface-name "$IFACE" \
        connection.autoconnect yes \
        connection.autoconnect-priority "$AUTOCONNECT_PRIORITY" \
        ipv4.method manual \
        ipv4.addresses "$LOCAL_CIDR" \
        ipv4.gateway "" \
        ipv4.routes "" \
        ipv4.dns "" \
        ipv4.ignore-auto-dns yes \
        ipv4.never-default yes \
        ipv6.method disabled \
        ipv6.gateway "" \
        ipv6.routes "" \
        ipv6.dns "" \
        ipv6.ignore-auto-dns yes \
        ipv6.never-default yes \
        || die "cannot modify '$PROFILE' (NetworkManager authorization denied)"
    check_profile
    ok "profile prepared but not activated; run '$0 activate' when ready"
}

activate_profile() {
    need_commands
    check_profile
    local active
    active="$(nmcli -g GENERAL.CONNECTION device show "$IFACE" 2>/dev/null || true)"
    if [[ "$active" != "$PROFILE" ]]; then
        management_guard
        echo "[g1-network] activating '$PROFILE' on $IFACE (no sudo/password prompt)"
        nmcli connection up id "$PROFILE" ifname "$IFACE" \
            || die "activation denied/failed; no fallback route or DNS change was attempted"
    else
        ok "'$PROFILE' is already active on $IFACE"
    fi
    check_contract
}

show_status() {
    need_commands
    echo "profile configured: $(nmcli -g connection.interface-name connection show "$PROFILE" 2>/dev/null || echo '<missing>')"
    echo "active on $IFACE : $(nmcli -g GENERAL.CONNECTION device show "$IFACE" 2>/dev/null || echo '<unknown>')"
    echo "IPv4 on $IFACE   : $(ip -o -4 address show dev "$IFACE" scope global 2>/dev/null | awk '{print $4}' | paste -sd, -)"
    echo "route to G1      : $(route_line "$ROBOT_IP" 2>/dev/null || echo '<none>')"
    echo "route to WM      : $(route_line "$WM_IP" 2>/dev/null || echo '<none>')"
    echo "DNS on $IFACE    : $(resolvectl dns "$IFACE" 2>/dev/null || echo '<unknown>')"
    echo "resolver default : $(resolvectl default-route "$IFACE" 2>/dev/null || echo '<unknown>')"
}

case "${1:-}" in
    check) check_contract "${2:-}" ;;
    activate) activate_profile ;;
    prepare) prepare_profile ;;
    status) show_status ;;
    -h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
