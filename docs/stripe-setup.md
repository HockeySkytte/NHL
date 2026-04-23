# Stripe Setup

This app uses server-side Stripe Checkout and the Stripe Billing Portal.
The browser never needs the Stripe secret key.

## What You Need In Stripe

1. Create one product for Pro access.
2. Create two recurring prices under that product:
   - Monthly: `$5/month`
   - Yearly: `$40/year`
3. Enable the Billing Portal in the Stripe dashboard.
4. Add a webhook endpoint that points to:
   - Local: `http://localhost:5000/stripe/webhook`
   - Production: `https://<your-domain>/stripe/webhook`
5. Subscribe the webhook to these events:
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`

## Environment Variables

Set these in your local `.env` and in your production host secret store.
Do not commit real values into the repo.

```env
STRIPE_SECRET_KEY=sk_test_or_sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_MONTHLY_ID=price_...
STRIPE_PRICE_YEARLY_ID=price_...
APP_BASE_URL=https://your-site.example
```

Notes:
- `STRIPE_SECRET_KEY` stays server-side only.
- `STRIPE_WEBHOOK_SECRET` is only used to verify webhook signatures.
- `APP_BASE_URL` is used to build return URLs safely behind a proxy.
- A publishable key is not required for the current implementation because the app redirects to Stripe Checkout from the server.

## Local Testing

1. Use Stripe test mode.
2. Put the test secret key and test price IDs in `.env`.
3. Start the Flask app.
4. Run Stripe CLI forwarding:

```bash
stripe listen --forward-to http://localhost:5000/stripe/webhook
```

5. Copy the webhook signing secret from the CLI output into `STRIPE_WEBHOOK_SECRET`.
6. Open the Account page and start checkout from there.

## Production Setup

1. Create live prices in Stripe live mode.
2. Add the live env vars in Render or your production secret manager.
3. Configure the production webhook endpoint in Stripe.
4. Do not store any Stripe secrets in templates, JavaScript, or committed files.

## Security Rules

1. Never expose `STRIPE_SECRET_KEY` or `STRIPE_WEBHOOK_SECRET` in HTML, JS, or client-side API responses.
2. Keep secrets only in `.env`, Render environment variables, or another server-side secret manager.
3. Do not add Supabase RLS policies that let normal users update `subscription_status`, `subscription_plan`, or `billing_interval`.
4. Let Stripe webhooks update billing state after checkout instead of trusting browser form input.
5. Keep the Supabase service key server-side only.

## How Billing State Flows

1. The account page posts the selected plan to Flask.
2. Flask creates a Stripe Checkout session using the secret key.
3. Stripe redirects the user back after checkout.
4. Stripe sends webhook events to `/stripe/webhook`.
5. The webhook updates `public.user_accounts` with the Stripe customer and subscription IDs plus the current plan state.
6. The app reads `user_accounts` to decide whether Projections access is allowed.

## Database Changes

Apply these migrations in Supabase SQL Editor or your migration workflow:
- `supabase/migrations/010_harden_user_accounts_constraints.sql`
- `supabase/migrations/011_add_user_account_stripe_fields.sql`

## If Something Does Not Update

Check these first:
- The webhook endpoint is reachable.
- The webhook is subscribed to the four required events.
- `STRIPE_WEBHOOK_SECRET` matches the endpoint or Stripe CLI forwarding secret.
- The selected price IDs are from the same Stripe mode as the secret key.
- `APP_BASE_URL` matches the real site URL.
