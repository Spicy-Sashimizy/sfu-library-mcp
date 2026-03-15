# WeChat Official Account Setup Guide

## Overview

WeChat Official Account (微信公众号) is a business presence on WeChat that allows publishing articles, responding to inquiries, and building a subscriber base. This is critical for reaching the Chinese-speaking construction community in Greater Vancouver.

## Account Types

| Type | Verification | API Access | Cost |
|------|-------------|------------|------|
| Subscription Account (订阅号) | Optional | Limited | Free |
| Service Account (服务号) | Required | Full API | Free + verification fee |

**Recommendation:** Start with a **Service Account** for full API access and credibility.

## Registration Steps

### 1. Prerequisites

- Business license (BC business registration)
- Government-issued ID of account administrator
- Company bank account
- Corporate email address
- Mobile phone number linked to a WeChat account

### 2. Create the Account

1. Go to [mp.weixin.qq.com](https://mp.weixin.qq.com)
2. Click "Register Now" (立即注册)
3. Select "Service Account" (服务号)
4. Enter email address (must not be linked to another WeChat account)
5. Verify email
6. Select account region: **Canada**
7. Fill in organization details:
   - Organization name: Vanse Industry / 万思工业
   - Business license number
   - Administrator name and ID

### 3. Verification

Verification adds a blue checkmark and enables full API access.

- **Fee:** $99 USD (or ~300 RMB), paid annually
- **Process:**
  1. Submit business documents through the WeChat platform
  2. WeChat's third-party auditor reviews documents (3-5 business days)
  3. Small verification deposit to corporate bank account
  4. Confirm deposit amount on the platform
- **Documents needed:**
  - Business registration certificate
  - Bank account verification letter
  - Administrator ID scan

### 4. Account Configuration

After verification:

1. **Set account name:** 万思工业 Vanse Industry
2. **Set account ID:** vanseindustry (used in search)
3. **Upload profile photo:** Company logo
4. **Set auto-reply:**
   - Welcome message for new followers
   - Keyword auto-replies for common equipment inquiries
5. **Configure menu:**
   - Equipment catalog link
   - Contact us
   - Website link (vanseindustry.com)

## API Access (After Verification)

### Capabilities

- Send template messages to subscribers
- Create custom menus
- Access user profiles (OpenID)
- Receive and respond to messages programmatically
- Generate QR codes for offline marketing

### Integration Notes

- API documentation: [developers.weixin.qq.com](https://developers.weixin.qq.com/doc/)
- Use `appid` and `appsecret` from the Official Account admin panel
- Access tokens expire every 2 hours — implement token refresh
- Message push: limited to 4 articles per month for Service Accounts
- Template messages: unlimited, but require pre-approved templates

## Content Posting Guidelines

### Frequency

- Service Account: maximum 4 pushes per month (each push can contain multiple articles)
- Recommended: 1 push per week (matches config.yaml setting)

### Content Types (from config)

1. **Equipment Showcase** — feature a specific equipment type with specs and pricing
2. **Promotion** — seasonal deals, clearance, special financing offers
3. **Company Intro** — Vanse Industry background, team, locations

### Best Practices

- Post between 6-8 PM Pacific (peak WeChat usage for Vancouver Chinese community)
- Use high-quality equipment photos
- Include clear pricing in CAD
- Always include contact method (WeChat ID, phone)
- Use the content templates in `chinese_channels/wechat_content.py`

## QR Code Marketing

Generate WeChat QR codes for:
- Business cards
- Equipment tags at warehouse
- Flyers at construction sites
- Trade show materials

QR codes can be generated in the Official Account admin panel under "Account Settings."

## Cost Summary

| Item | Cost | Frequency |
|------|------|-----------|
| Account registration | Free | One-time |
| Verification | $99 USD | Annual |
| Content creation | Internal | Ongoing |
| **Total Year 1** | **$99** | |
