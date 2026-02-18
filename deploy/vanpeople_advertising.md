# VanPeople Advertising Evaluation

## Overview

VanPeople (人在温哥华) is one of the largest Chinese-language community platforms in Greater Vancouver. It offers classified listings, forums, and advertising options targeting the local Chinese community.

## Current Strategy: Organic Monitoring

The `scrapers/vanpeople_classifieds.py` scraper monitors equipment-related listings on VanPeople and extracts:
- WTB (求购) signals — potential buyers looking for equipment
- WTS (出售) signals — competitors or potential trade-in leads
- Contact info (WeChat IDs, phone numbers)
- Equipment types and price points

This organic monitoring runs daily and costs nothing.

## Paid Advertising Options

### 1. Featured Listing (置顶帖)

- **What:** Pin a classified ad to the top of a category
- **Cost:** ~$50-150 CAD per week (varies by category)
- **Reach:** High visibility within the construction/equipment category
- **Best for:** Advertising specific equipment for sale

### 2. Banner Advertising

- **What:** Display banner on category pages or homepage
- **Cost:** ~$300-800 CAD per month (varies by placement)
- **Reach:** Broad exposure across all VanPeople visitors
- **Best for:** Brand awareness for Vanse Industry

### 3. Category Sponsorship

- **What:** Sponsored presence in the construction equipment category
- **Cost:** Negotiable, typically $500-1000 CAD per month
- **Reach:** All visitors browsing construction equipment
- **Best for:** Establishing authority in the equipment niche

### 4. Forum Sponsored Posts

- **What:** Promoted posts in relevant forums (construction, business, jobs)
- **Cost:** ~$100-300 CAD per post
- **Reach:** Active community members
- **Best for:** Long-form content, equipment guides, company introductions

## Expected Reach

- VanPeople monthly visitors: ~200,000-300,000 (Greater Vancouver Chinese community)
- Construction/equipment category: ~5,000-10,000 monthly views (estimated)
- Target demographic overlap: moderate (construction workers, small business owners)

## ROI Analysis

### Organic Monitoring (Current)

| Metric | Value |
|--------|-------|
| Monthly cost | $0 |
| Leads captured | 5-15 per month (WTB signals) |
| Conversion rate | ~5-10% (estimated) |
| Expected sales | 0.5-1.5 per month |

### With Paid Featured Listings

| Metric | Value |
|--------|-------|
| Monthly cost | ~$200-600 CAD |
| Additional leads | 10-30 per month (estimated) |
| Conversion rate | ~3-5% (lower for ads vs organic intent) |
| Expected additional sales | 0.3-1.5 per month |
| Breakeven | ~1 equipment sale per quarter |

### With Banner + Featured

| Metric | Value |
|--------|-------|
| Monthly cost | ~$500-1,400 CAD |
| Leads (direct + brand) | 20-50 per month |
| Brand awareness lift | Significant in Chinese community |
| Breakeven | ~1-2 equipment sales per quarter |

## Recommendation

**Start with organic monitoring only. Evaluate paid after 30 days of data.**

### Phase 1 (Current — Month 1-2)
- Run vanpeople_classifieds scraper daily
- Track WTB signals and conversion rates
- Build baseline metrics

### Phase 2 (Month 3 — If Organic Shows Promise)
- Post 1-2 organic classified ads per week (free)
- Test response rates from classified ads vs scraped leads
- Collect data on which equipment types get the most interest

### Phase 3 (Month 4+ — If ROI Justifies)
- Start with featured listings ($50-150/week)
- A/B test: featured listing vs organic-only weeks
- Track cost per lead and cost per sale

### Phase 4 (Month 6+ — Scale If Profitable)
- Consider banner advertising for brand awareness
- Evaluate category sponsorship
- Compare VanPeople ROI vs other Chinese channels (WeChat, XiaoHongShu)

## VanSky Comparison

VanSky (温哥华天空) is similar to VanPeople but with a smaller user base. The same organic monitoring strategy applies via `scrapers/vansky_classifieds.py`. Paid advertising on VanSky is generally cheaper but with proportionally lower reach.

**Recommendation:** Focus paid advertising budget on VanPeople first due to larger audience. Use VanSky for organic monitoring only unless VanPeople ROI is proven.

## Config Reference

The `config/config.yaml` tracks paid listing status:

```yaml
chinese_channels:
  platforms:
    vanpeople:
      paid_listings: false  # Set to true when paid ads activated
      organic_monitoring: true
    vansky:
      paid_listings: false
      organic_monitoring: true
```
