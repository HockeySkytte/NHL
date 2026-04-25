import os

import pytest

from app import create_app
import app.routes as routes


@pytest.fixture(scope='module')
def app_instance():
    os.environ['XG_PRELOAD'] = '0'
    os.environ['PRESTART_LOGGER'] = '0'
    app = create_app()
    app.config.update(TESTING=True)
    return app


@pytest.fixture()
def client(app_instance):
    return app_instance.test_client()


def test_about_redirect(client):
    response = client.get('/about', follow_redirects=False)
    assert response.status_code in (301, 302, 307, 308)
    assert '/about/inside-the-app' in (response.headers.get('Location') or '')


def test_about_section_page(client):
    response = client.get('/about/how-to-use-the-app')
    assert response.status_code == 200
    assert b'How to Use the App' in response.data


def test_about_glossary_page(client):
    response = client.get('/about/glossary')
    assert response.status_code == 200
    assert b'Glossary' in response.data


def test_account_requires_login(client):
    response = client.get('/account', follow_redirects=False)
    assert response.status_code in (301, 302, 307, 308)
    assert '/login' in (response.headers.get('Location') or '')


def test_admin_run_update_data_requires_admin(client):
    response = client.post('/admin/run-update-data', json={'date': '2026-04-24'})
    assert response.status_code in (401, 403)


def test_admin_run_update_data_validation_for_admin(monkeypatch, client):
    admin_user = {
        'user_id': 'admin-1',
        'is_admin': True,
        'has_access': True,
        'email': 'admin@example.com',
    }
    monkeypatch.setattr(routes, '_refresh_current_auth_user', lambda: admin_user)
    monkeypatch.setattr(routes, '_current_auth_user', lambda: admin_user)
    monkeypatch.setattr(routes, '_start_admin_job', lambda command, cwd: 'test-job-1')

    bad = client.post('/admin/run-update-data', data='not-json', content_type='text/plain')
    assert bad.status_code == 400

    ok = client.post(
        '/admin/run-update-data',
        json={'date': '2026-04-24', 'export': False, 'projections_to_sheets': False},
    )
    assert ok.status_code == 200
    data = ok.get_json() or {}
    assert data.get('jobId') == 'test-job-1'


def test_account_post_requires_csrf(monkeypatch, client):
    auth_user = {
        'user_id': 'user-1',
        'is_admin': False,
        'has_access': True,
        'email': 'user@example.com',
        'username': 'user1',
    }
    monkeypatch.setattr(routes, '_refresh_current_auth_user', lambda: auth_user)
    monkeypatch.setattr(routes, '_current_auth_user', lambda: auth_user)
    monkeypatch.setattr(routes, '_sb_get_user_account', lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, '_find_user_account_by_username', lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, '_persist_auth_user_updates', lambda *args, **kwargs: auth_user)

    with client.session_transaction() as sess:
        sess[routes._AUTH_SESSION_KEY] = auth_user
        sess[routes._CSRF_SESSION_KEY] = 'csrf-ok'

    bad = client.post('/account/profile', data={'username': 'newuser'})
    assert bad.status_code == 400

    good = client.post('/account/profile', data={'username': 'newuser', 'csrf_token': 'csrf-ok'}, follow_redirects=False)
    assert good.status_code in (301, 302, 303, 307, 308)


def test_admin_form_post_requires_csrf(monkeypatch, client):
    admin_user = {
        'user_id': 'admin-1',
        'is_admin': True,
        'has_access': True,
        'email': 'admin@example.com',
    }
    monkeypatch.setattr(routes, '_refresh_current_auth_user', lambda: admin_user)
    monkeypatch.setattr(routes, '_current_auth_user', lambda: admin_user)
    monkeypatch.setattr(routes, '_normalize_username', lambda value: str(value or '').strip().lower())
    monkeypatch.setattr(routes, '_valid_username', lambda value: True)
    monkeypatch.setattr(routes, '_valid_email', lambda value: True)
    monkeypatch.setattr(routes, '_find_user_account_by_username', lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, '_find_user_account_by_email', lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, '_find_auth_user_by_email', lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, '_sb_auth_admin_create_user', lambda *args, **kwargs: {'user': {'id': 'u-1', 'email': 'new@example.com'}})
    monkeypatch.setattr(routes, '_auth_record_from_supabase_user', lambda *args, **kwargs: {'user_id': 'u-1', 'email': 'new@example.com'})
    monkeypatch.setattr(routes, '_subscription_update_for_plan', lambda *args, **kwargs: {})
    monkeypatch.setattr(routes, '_build_account_payload', lambda *args, **kwargs: {})
    monkeypatch.setattr(routes, '_sb_upsert_user_account', lambda *args, **kwargs: {})

    with client.session_transaction() as sess:
        sess[routes._AUTH_SESSION_KEY] = admin_user
        sess[routes._CSRF_SESSION_KEY] = 'csrf-admin'

    form_data = {
        'username': 'newuser',
        'email': 'new@example.com',
        'password': 'password123',
        'confirm_password': 'password123',
        'access': 'trial',
    }

    bad = client.post('/admin/users/create', data=form_data)
    assert bad.status_code == 400

    form_data['csrf_token'] = 'csrf-admin'
    good = client.post('/admin/users/create', data=form_data, follow_redirects=False)
    assert good.status_code in (301, 302, 303, 307, 308)


def test_account_plan_stripe_failure_is_handled(monkeypatch, client):
    auth_user = {
        'user_id': 'user-1',
        'is_admin': False,
        'has_access': True,
        'email': 'user@example.com',
        'username': 'user1',
        'stripe_subscription_id': '',
    }
    monkeypatch.setattr(routes, '_refresh_current_auth_user', lambda: auth_user)
    monkeypatch.setattr(routes, '_current_auth_user', lambda: auth_user)
    monkeypatch.setattr(routes, '_stripe_any_configured', lambda: True)
    monkeypatch.setattr(routes, '_stripe_missing_config', lambda plan_key=None: [])
    monkeypatch.setattr(routes, '_stripe_price_id', lambda _plan_key: 'price_test_123')

    class _CheckoutSessionApi:
        @staticmethod
        def create(**kwargs):
            raise RuntimeError('stripe unavailable')

    class _CheckoutApi:
        Session = _CheckoutSessionApi

    class _StripeClient:
        checkout = _CheckoutApi

    monkeypatch.setattr(routes, '_stripe_client', lambda: _StripeClient())

    with client.session_transaction() as sess:
        sess[routes._AUTH_SESSION_KEY] = auth_user
        sess[routes._CSRF_SESSION_KEY] = 'csrf-ok'

    response = client.post(
        '/account/plan',
        data={'plan': 'monthly', 'csrf_token': 'csrf-ok'},
        follow_redirects=False,
    )
    assert response.status_code in (301, 302, 303, 307, 308)
    assert '/account' in (response.headers.get('Location') or '')
