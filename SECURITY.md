# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of nadoo-flow-core seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please DO NOT:

- Open a public GitHub issue for security vulnerabilities
- Disclose the vulnerability publicly before we have released a fix
- Exploit the vulnerability beyond what is necessary to demonstrate it

### Please DO:

1. **Email us directly** at **security@nadoo.ai**
2. Include a detailed description of the vulnerability
3. Provide steps to reproduce the issue
4. Include any proof-of-concept code (if applicable)
5. Suggest a fix or mitigation (if you have one)

### What to Include in Your Report

A good security report should include:

- **Description**: What is the vulnerability?
- **Impact**: What can an attacker do with this vulnerability?
- **Affected Versions**: Which versions are affected?
- **Reproduction Steps**: How to reproduce the vulnerability?
- **Proof of Concept**: Code demonstrating the issue (optional but helpful)
- **Suggested Fix**: Your ideas for fixing it (optional)

Example:
```
Subject: [SECURITY] Potential code injection in custom nodes

Description:
User-provided code in custom nodes may allow arbitrary code execution
if not properly sandboxed.

Impact:
An attacker could execute malicious code on the server by creating
a malicious custom node.

Affected Versions:
0.1.0 and earlier

Steps to Reproduce:
1. Create a custom node with malicious code
2. Execute workflow
3. Observe code execution

Suggested Fix:
Implement proper sandboxing using RestrictedPython or similar.
```

## Response Timeline

- **Initial Response**: Within 48 hours
- **Status Updates**: Every 72 hours
- **Resolution Timeline**: Depends on severity
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 90 days

## Disclosure Policy

Once we have a fix ready:

1. We will notify you
2. We will coordinate disclosure timing with you
3. We will release a security advisory
4. We will credit you in the advisory (unless you prefer to remain anonymous)
5. We will release a patched version

## Security Best Practices

When using nadoo-flow-core in your applications:

### 1. Input Validation

Always validate user inputs in custom nodes:

```python
from nadoo_flow import ChainableNode, NodeResult

class MyNode(ChainableNode):
    async def execute(self, node_context, workflow_context):
        user_input = node_context.input_data.get("user_input", "")

        # Validate input
        if not isinstance(user_input, str):
            raise ValueError("Invalid input type")
        if len(user_input) > 10000:
            raise ValueError("Input too long")

        # Process safely
        return NodeResult(success=True, output={"result": user_input.strip()})
```

### 2. Dependency Management

- Keep nadoo-flow-core updated to the latest version
- Regularly audit your dependencies with `pip-audit` or similar tools
- Use virtual environments to isolate dependencies

```bash
# Check for vulnerabilities
pip install pip-audit
pip-audit

# Update nadoo-flow-core
pip install --upgrade nadoo-flow-core
```

### 3. Backend Selection

When using custom backends:

- Only use trusted, well-maintained backends
- Review backend source code before deployment
- Monitor backend dependencies for vulnerabilities
- Use the native backend for maximum control

```python
from nadoo_flow import BackendRegistry

# Use trusted backends only
backend = BackendRegistry.create("native")  # Recommended

# Review before using third-party backends
# backend = BackendRegistry.create("third-party")  # Review carefully
```

### 4. Code Execution

If allowing user-provided code execution:

- Use sandboxing (e.g., RestrictedPython)
- Set resource limits (CPU, memory, time)
- Run in isolated containers
- Never run with elevated privileges

### 5. Network Security

- Use HTTPS for all external API calls
- Validate SSL certificates
- Don't expose internal services to public networks
- Use API keys/tokens securely (environment variables, not hardcoded)

```python
import os
from nadoo_flow import ChainableNode

class APINode(ChainableNode):
    async def execute(self, node_context, workflow_context):
        # Good: Use environment variables
        api_key = os.environ.get("API_KEY")

        # Bad: Hardcoded credentials
        # api_key = "secret-key-123"  # DON'T DO THIS

        # ... make API call ...
```

### 6. Logging and Monitoring

- Don't log sensitive data (passwords, API keys, PII)
- Monitor for unusual patterns
- Set up alerts for errors/failures
- Audit workflow execution logs

### 7. Production Deployment

- Use production-grade web servers (not development servers)
- Enable HTTPS/TLS
- Set up proper authentication and authorization
- Use rate limiting
- Implement CORS policies
- Keep logs secure and rotated

## Known Security Considerations

### 1. Custom Node Execution

Custom nodes execute arbitrary Python code. This is by design but requires careful consideration:

- **Risk**: Malicious code in custom nodes
- **Mitigation**: Only load trusted nodes, use code review, consider sandboxing

### 2. Backend Plugins

Third-party backends may have their own security considerations:

- **Risk**: Vulnerabilities in backend implementations
- **Mitigation**: Review backend code, keep backends updated, use native backend when possible

### 3. Workflow Serialization

Workflows may be serialized/deserialized:

- **Risk**: Code injection via malicious workflow definitions
- **Mitigation**: Validate workflow structure, sanitize inputs, use schema validation

## Security Checklist

Before deploying nadoo-flow-core in production:

- [ ] All dependencies are up to date
- [ ] Security patches are applied
- [ ] Input validation is implemented
- [ ] Code execution is sandboxed (if applicable)
- [ ] Secrets are managed securely (not in code)
- [ ] HTTPS/TLS is enabled
- [ ] Authentication is implemented
- [ ] Authorization is enforced
- [ ] Logging excludes sensitive data
- [ ] Monitoring and alerts are set up
- [ ] Backup and recovery plan is in place

## Contact

- **Security Email**: security@nadoo.ai
- **General Contact**: dev@nadoo.ai
- **GitHub**: https://github.com/nadoo-ai/nadoo-flow-core/security/advisories

## Credits

We thank the security researchers who have responsibly disclosed vulnerabilities:

- (No vulnerabilities reported yet)

## Updates

This security policy may be updated from time to time. Please check back regularly.

**Last Updated**: 2025-01-22
