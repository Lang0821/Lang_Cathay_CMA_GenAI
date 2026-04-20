from __future__ import annotations

import argparse

from workflow import list_profiles, run_finance_copilot


def main() -> None:
    parser = argparse.ArgumentParser(description="CUBE Copilot CLI Demo")
    parser.add_argument("--profile", default="young_professional", help="Profile ID")
    parser.add_argument(
        "--question",
        default="我想在 6 個月內存到 10 萬，但最近餐飲和購物有點超支，該怎麼安排？",
        help="User question",
    )
    args = parser.parse_args()

    valid_profiles = {item["id"] for item in list_profiles()}
    if args.profile not in valid_profiles:
        raise SystemExit(f"Unknown profile. Available: {', '.join(sorted(valid_profiles))}")

    result = run_finance_copilot(profile_id=args.profile, user_query=args.question)
    print("=" * 80)
    print(result["final_response"])
    print("=" * 80)
    if result.get("warnings"):
        print("Warnings:")
        for item in result["warnings"]:
            print(f"- {item}")
    if result.get("sources"):
        print("Sources:")
        for item in result["sources"]:
            print(f"- {item}")


if __name__ == "__main__":
    main()
