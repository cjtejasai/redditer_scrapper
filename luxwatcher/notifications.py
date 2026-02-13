"""
Email notifications using SendGrid.
"""
from __future__ import annotations

import os
from typing import Any

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Cc, Content


def send_leads_notification(
    leads_count: int,
    total_count: int,
    run_id: str,
    leads_data: list[dict] | None = None,
    dashboard_url: str | None = None,
) -> dict[str, Any]:
    """
    Send email notification after a successful pipeline run.

    Returns:
        dict with 'success' (bool) and 'message' or 'error'
    """
    api_key = os.getenv("SENDGRID_API_KEY")
    if not api_key:
        return {"success": False, "error": "SENDGRID_API_KEY not configured"}

    from_email = os.getenv("SENDGRID_FROM_EMAIL", "noreply@darglobal.co.uk")
    to_emails_str = os.getenv("SENDGRID_TO_EMAILS", "")

    if not to_emails_str:
        return {"success": False, "error": "SENDGRID_TO_EMAILS not configured"}

    to_emails = [e.strip() for e in to_emails_str.split(",") if e.strip()]
    if not to_emails:
        return {"success": False, "error": "No valid recipient emails"}

    # Get CC emails
    cc_emails_str = os.getenv("SENDGRID_CC_EMAILS", "")
    cc_emails = [e.strip() for e in cc_emails_str.split(",") if e.strip()]

    # Build email content
    subject = f"Reddit Watcher: {leads_count} New Leads Found"

    # Build HTML content - comprehensive table format
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.5; color: #333; margin: 0; padding: 0; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
            .header h1 {{ margin: 0; font-size: 26px; font-weight: 600; }}
            .header p {{ margin: 10px 0 0 0; opacity: 0.9; font-size: 16px; }}
            .content {{ padding: 25px; background: #fff; border-radius: 0 0 8px 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .table-wrapper {{ overflow-x: auto; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
            th {{ background: #1a1a2e; color: white; padding: 12px 10px; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; white-space: nowrap; }}
            td {{ padding: 12px 10px; border-bottom: 1px solid #eee; vertical-align: top; }}
            tr:nth-child(even) {{ background: #fafafa; }}
            tr:hover {{ background: #f0f7ff; }}
            .confidence-high {{ background: #e8f5e9; color: #2e7d32; padding: 4px 8px; border-radius: 4px; font-weight: bold; display: inline-block; }}
            .confidence-medium {{ background: #fff3e0; color: #e65100; padding: 4px 8px; border-radius: 4px; font-weight: bold; display: inline-block; }}
            .confidence-low {{ background: #ffebee; color: #c62828; padding: 4px 8px; border-radius: 4px; font-weight: bold; display: inline-block; }}
            .stage-ready {{ background: #e8f5e9; color: #2e7d32; padding: 3px 8px; border-radius: 4px; font-size: 11px; }}
            .stage-active {{ background: #e3f2fd; color: #1565c0; padding: 3px 8px; border-radius: 4px; font-size: 11px; }}
            .stage-research {{ background: #f3e5f5; color: #7b1fa2; padding: 3px 8px; border-radius: 4px; font-size: 11px; }}
            .source-op {{ background: #fff3e0; color: #e65100; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; }}
            .source-commenter {{ background: #e0f2f1; color: #00695c; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; }}
            .link {{ color: #1a73e8; text-decoration: none; font-weight: 500; }}
            .link:hover {{ text-decoration: underline; }}
            .evidence {{ font-style: italic; color: #666; font-size: 12px; max-width: 250px; }}
            .reason {{ color: #555; font-size: 12px; max-width: 200px; }}
            .btn {{ display: inline-block; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 14px 35px; text-decoration: none; border-radius: 6px; margin-top: 25px; font-weight: 600; }}
            .footer {{ text-align: center; padding: 20px; color: #888; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Reddit Watcher Report</h1>
                <p>{leads_count} New Leads Found</p>
            </div>
            <div class="content">
    """

    # Add leads table if available
    if leads_data and leads_count > 0:
        html_content += """
                <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Summary</th>
                            <th>Source</th>
                            <th>Stage</th>
                            <th>Property</th>
                            <th>Budget</th>
                            <th>Confidence</th>
                            <th>Evidence</th>
                            <th>Reason</th>
                            <th>Link</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for i, lead in enumerate(leads_data[:20], 1):  # Show up to 20 leads
            url = lead.get("url", "")
            confidence = float(lead.get("confidence", 0))
            property_type = lead.get("property_type", "unknown").replace("_", " ").title()
            buyer_stage = lead.get("buyer_stage", "researching")
            budget = lead.get("budget", "not_mentioned")
            source = lead.get("source", "op")
            evidence = lead.get("evidence", "")[:150]
            reason = lead.get("reason", "")[:120]
            thread_summary = lead.get("thread_summary", "")[:100] or "View thread"

            if budget == "not_mentioned":
                budget = "-"

            # Confidence styling
            if confidence >= 0.7:
                conf_class = "confidence-high"
            elif confidence >= 0.4:
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"

            # Stage styling
            stage_display = buyer_stage.replace("_", " ").title()
            if buyer_stage == "ready_to_buy":
                stage_class = "stage-ready"
            elif buyer_stage == "actively_looking":
                stage_class = "stage-active"
            else:
                stage_class = "stage-research"

            # Source styling
            source_display = "OP" if source == "op" else "Commenter"
            source_class = "source-op" if source == "op" else "source-commenter"

            html_content += f"""
                        <tr>
                            <td><strong>{i}</strong></td>
                            <td style="max-width: 200px;">{thread_summary}</td>
                            <td><span class="{source_class}">{source_display}</span></td>
                            <td><span class="{stage_class}">{stage_display}</span></td>
                            <td>{property_type}</td>
                            <td><strong>{budget}</strong></td>
                            <td><span class="{conf_class}">{confidence:.0%}</span></td>
                            <td class="evidence">"{evidence}"</td>
                            <td class="reason">{reason}</td>
                            <td><a href="{url}" class="link" target="_blank">View</a></td>
                        </tr>
            """

        html_content += """
                    </tbody>
                </table>
                </div>
        """
    else:
        html_content += """
                <p style="text-align: center; color: #666; padding: 40px;">No leads found in this run.</p>
        """

    # Add dashboard link if available
    if dashboard_url:
        html_content += f"""
                <div style="text-align: center;">
                    <a href="{dashboard_url}" class="btn">View Full Dashboard</a>
                </div>
        """

    html_content += """
            </div>
            <div class="footer">
                Reddit Watcher - Automated Lead Detection
            </div>
        </div>
    </body>
    </html>
    """

    try:
        sg = SendGridAPIClient(api_key)

        message = Mail(
            from_email=Email(from_email),
            to_emails=[To(email) for email in to_emails],
            subject=subject,
            html_content=Content("text/html", html_content),
        )

        # Add CC recipients
        if cc_emails:
            for cc_email in cc_emails:
                message.add_cc(Cc(cc_email))

        response = sg.send(message)

        return {
            "success": True,
            "message": f"Email sent to {len(to_emails)} recipients, CC: {len(cc_emails)}",
            "status_code": response.status_code,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
        }
