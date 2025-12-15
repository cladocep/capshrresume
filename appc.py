import streamlit as st
import pandas as pd
from agentcpsc import run_agent
import tiktoken
from datetime import datetime
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

st.set_page_config(page_title="HR Resume Chatbot", page_icon=":robot:", layout="wide")

# PDF Generation helper
def generate_candidate_pdf(candidate_data: dict) -> bytes:
    """Generate PDF dari candidate data."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Container untuk elements
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#2c5aa0'),
        spaceAfter=8,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        leading=14
    )
    
    # Title
    elements.append(Paragraph("RESUME KANDIDAT", title_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Metadata table
    metadata = [
        ['Informasi', 'Detail'],
        ['ID Kandidat', str(candidate_data.get('id', 'N/A'))],
        ['Kategori', str(candidate_data.get('category', 'Unknown'))],
        ['Score', f"{candidate_data.get('score', 0):.2f}"],
        ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
    ]
    
    metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    elements.append(metadata_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Content section
    elements.append(Paragraph("RESUME CONTENT", heading_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Resume text
    resume_text = candidate_data.get('text', 'No content available')
    # Split text into paragraphs untuk better formatting
    paragraphs = resume_text.split('\n')
    for para in paragraphs[:100]:  # Limit to prevent huge PDFs
        if para.strip():
            elements.append(Paragraph(para.strip(), body_style))
    
    if len(paragraphs) > 100:
        elements.append(Paragraph("[... Content truncated for PDF ...]", body_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# Token counting helpers
def count_tokens(text: str, model: str = "gpt-4-mini") -> int:
    """Hitung jumlah tokens dalam teks menggunakan tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: estimasi 1 token ≈ 4 karakter
        return len(text) // 4

def calculate_cost_idr(input_tokens: int, output_tokens: int, model: str = "gpt-4-mini") -> float:
    """
    Hitung estimasi biaya dalam IDR untuk OpenAI API.
    - gpt-4-mini: $0.15 per 1M input tokens, $0.6 per 1M output tokens
    - Kurs: 1 USD ≈ 17,000 IDR
    """
    # Harga per token (dalam USD)
    if "gpt-4-mini" in model or "gpt-4o-mini" in model:
        input_price_per_1m = 0.15
        output_price_per_1m = 0.6
    else:
        # Default ke gpt-4-mini pricing
        input_price_per_1m = 0.15
        output_price_per_1m = 0.6
    
    # Hitung biaya dalam USD
    input_cost_usd = (input_tokens / 1_000_000) * input_price_per_1m
    output_cost_usd = (output_tokens / 1_000_000) * output_price_per_1m
    total_cost_usd = input_cost_usd + output_cost_usd
    
    # Konversi ke IDR (17,000 IDR per USD)
    cost_idr = total_cost_usd * 17_000
    
    return cost_idr

def export_candidate_to_txt(candidate_data: dict) -> str:
    """Export candidate data ke format text yang rapi."""
    text = f"""
{'='*80}
RESUME KANDIDAT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ID: {candidate_data.get('id', 'N/A')}
Category: {candidate_data.get('category', 'Unknown')}
Score: {candidate_data.get('score', 'N/A')}

{'-'*80}
CONTENT:
{'-'*80}

{candidate_data.get('text', 'No content available')}

{'='*80}
"""
    return text

# State untuk chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "tool_logs" not in st.session_state:
    st.session_state.tool_logs = []

if "usage_logs" not in st.session_state:
    st.session_state.usage_logs = []

if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None

tab_chat, tab_logs, tab_usage = st.tabs(["Chat", "Tool Calls", "Usage Logs"])

# Sidebar untuk kandidat tools
st.sidebar.markdown("## 🔍 Candidate Tools")
st.sidebar.markdown("---")

# Lookup candidate by ID
st.sidebar.subheader("Lookup Candidate")
st.sidebar.caption("📝 Enter Candidate ID from Resume database")
candidate_id = st.sidebar.number_input("Candidate ID:", min_value=0, step=1, key="candidate_id_input")

if st.sidebar.button("Search by ID", key="lookup_btn"):
    try:
        from qdrant_client import QdrantClient
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        qdrant = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
        
        # Validate ID range
        if int(candidate_id) < 0:
            st.sidebar.error(f"⚠️ ID cannot be negative. You entered: {int(candidate_id)}")
        else:
            # Retrieve candidate by ID
            points = qdrant.retrieve('resume_embeddings', ids=[int(candidate_id)], with_payload=True)
            
            if points:
                point = points[0]
                candidate_data = {
                    'id': point.id,
                    'category': point.payload.get('category', point.payload.get('Category', 'Unknown')) if point.payload else 'Unknown',
                    'text': point.payload.get('text', '') if point.payload else '',
                    'score': 1.0
                }
                st.session_state.selected_candidate = candidate_data
                st.sidebar.success(f"✓ Found candidate #{int(candidate_id)}")
            else:
                st.sidebar.error(f"✗ Candidate #{int(candidate_id)} not found in database")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)[:100]}")

# Display selected candidate
if st.session_state.selected_candidate:
    st.sidebar.markdown("### Selected Candidate")
    cand = st.session_state.selected_candidate
    st.sidebar.write(f"**ID:** {cand['id']}")
    st.sidebar.write(f"**Category:** {cand['category']}")
    st.sidebar.write(f"**Content Length:** {len(cand['text'])} chars")
    
    # Export buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        txt_export = export_candidate_to_txt(cand)
        st.download_button(
            label="📄 TXT",
            data=txt_export,
            file_name=f"candidate_{cand['id']}.txt",
            mime="text/plain",
            key=f"export_txt_{cand['id']}"
        )
    
    with col2:
        pdf_export = generate_candidate_pdf(cand)
        st.download_button(
            label="� PDF",
            data=pdf_export,
            file_name=f"candidate_{cand['id']}.pdf",
            mime="application/pdf",
            key=f"export_pdf_{cand['id']}"
        )
    
    # View full content
    if st.sidebar.button("View Full Content", key="view_content_btn"):
        st.sidebar.markdown("### Full Resume Content")
        st.sidebar.text_area("Content:", value=cand['text'], height=300, disabled=True, key="content_view")
    
    st.sidebar.markdown("---")

with tab_chat:
    st.title("HR Resume Chatbot 🤖")

    # tampilkan history chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # input user
    user_input = st.chat_input("Ketik pesan Anda di sini...")
    if user_input:
        # simpan dan tampilkan pesan user
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # panggil rag
        try:
            with st.chat_message("assistant"):
                with st.spinner("Memproses..."):
                    result = run_agent(user_input)

                    answer = result["answer"]
                    debug = result["debug"]
                    st.markdown(answer)
                    
                    # Hitung token usage
                    input_token_count = count_tokens(user_input)
                    output_token_count = count_tokens(answer)
                    total_tokens = input_token_count + output_token_count
                    est_cost = calculate_cost_idr(input_token_count, output_token_count)
                    
                    # Tampilkan metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Input Tokens", input_token_count)
                    col2.metric("Output Tokens", output_token_count)
                    col3.metric("Total Tokens", total_tokens)
                    col4.metric("Est. Cost (IDR)", f"Rp {est_cost:,.0f}")

            # simpan jawaban ke history
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # simpan log tools & usage
            st.session_state.tool_logs.append(debug)
            st.session_state.usage_logs.append({
                "query": debug["query"],
                "num_docs": debug["num_docs"],
                "input_tokens": input_token_count,
                "output_tokens": output_token_count,
                "total_tokens": total_tokens,
                "est_cost_idr": est_cost,
            })
        except Exception as e:
            st.error(f"Agent error: {e}")

with tab_logs:
    st.subheader("Tool Calls Logs")
    if not st.session_state.tool_logs:
        st.info("Belum ada tool calls yang tercatat. Coba tanya sesuatu di tab chat")
    else:
        for i, log in enumerate(reversed(st.session_state.tool_logs), start=1):
            with st.expander(f"Call #{i} - Query: {log.get('query', '-') }"):
                # Raw Document Objects
                st.markdown("#### 📋 Raw Document Objects from Retriever")
                if log.get('raw_documents'):
                    st.code(str(log.get('raw_documents')), language="python")
                else:
                    st.info("Raw documents not available in this log")
                
                st.divider()
                
                # Results Summary with PDF Export
                if log.get('doc_preview'):
                    st.markdown("#### 📊 Results Summary")
                    st.write(f"Jumlah kandidat ditemukan: {log.get('num_docs', 0)}")
                    
                    # Generate PDF dengan semua hasil
                    buffer = BytesIO()
                    pdf_doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
                    elements = []
                    styles = getSampleStyleSheet()
                    
                    title_style = ParagraphStyle(
                        'Title',
                        parent=styles['Heading1'],
                        fontSize=16,
                        textColor=colors.HexColor('#1f4788'),
                        spaceAfter=6,
                        alignment=TA_CENTER,
                        fontName='Helvetica-Bold'
                    )
                    
                    # Title
                    elements.append(Paragraph("SEARCH RESULTS", title_style))
                    elements.append(Spacer(1, 0.1*inch))
                    elements.append(Paragraph(f"Query: <b>{log.get('query')}</b>", styles['Normal']))
                    elements.append(Paragraph(f"Results: {len(log.get('doc_preview', []))} candidates", styles['Normal']))
                    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
                    elements.append(Spacer(1, 0.2*inch))
                    
                    # Add each candidate
                    for candidate_doc in log.get('doc_preview', []):
                        elements.append(Paragraph(f"KANDIDAT #{candidate_doc['idx']}", ParagraphStyle(
                            'CandTitle',
                            parent=styles['Heading2'],
                            fontSize=11,
                            textColor=colors.HexColor('#2c5aa0'),
                            spaceBefore=10,
                            spaceAfter=6,
                            fontName='Helvetica-Bold'
                        )))
                        elements.append(Paragraph(f"<b>Category:</b> {candidate_doc['category']}", styles['Normal']))
                        
                        # Add source and score info
                        source = candidate_doc.get('source', 'RAG Search')
                        score = candidate_doc.get('score', 0)
                        elements.append(Paragraph(f"<b>Source:</b> {source} | <b>Similarity Score:</b> {score}", styles['Normal']))
                        
                        elements.append(Paragraph(f"<b>Preview:</b><br/>{candidate_doc['snippet']}", styles['Normal']))
                        elements.append(Spacer(1, 0.15*inch))
                    
                    pdf_doc.build(elements)
                    buffer.seek(0)
                    pdf_data = buffer.getvalue()
                    
                    st.download_button(
                        label="📊 Export Results as PDF",
                        data=pdf_data,
                        file_name=f"search_results_{i}.pdf",
                        mime="application/pdf",
                        key=f"export_results_pdf_{i}"
                    )
                
                st.divider()
                
                # Retrieved Candidates Ranking
                st.markdown("#### 🎯 Retrieved Candidates Ranking")
                for idx, doc in enumerate(log.get("doc_preview", []), start=1):
                    col1, col2, col3 = st.columns([2.5, 0.8, 0.7])
                    with col1:
                        st.markdown(
                            f"**{idx}. ID #{doc['idx']}** | Category: `{doc['category']}`\n\n"
                            f"{doc['snippet']}"
                        )
                    with col2:
                        # Show source and score
                        source = doc.get('source', 'RAG Search')
                        score = doc.get('score', 0)
                        st.caption(f"📊 {source}\n⭐ {score}")
                    with col3:
                        # Add to selection button
                        if st.button("📌 Select", key=f"select_doc_{i}_{doc['idx']}"):
                            # Find full text in current result
                            candidate_data = {
                                'id': doc['idx'],
                                'category': doc['category'],
                                'text': doc['snippet'],
                                'score': doc.get('score', 0.9)
                            }
                            st.session_state.selected_candidate = candidate_data
                            st.success(f"Candidate {doc['idx']} selected!")

with tab_usage:
    st.subheader("Usage Details Logs")
    if not st.session_state.usage_logs:
        st.info("Belum ada usage logs yang tercatat. Coba tanya sesuatu di tab chat")
    else:
        df_usage = pd.DataFrame(st.session_state.usage_logs)
        
        # Format columns untuk tampilan lebih rapi
        if "est_cost_idr" in df_usage.columns:
            df_usage["est_cost_idr"] = df_usage["est_cost_idr"].apply(lambda x: f"Rp {x:,.0f}")
        
        st.dataframe(df_usage, use_container_width=True)

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_queries = len(st.session_state.usage_logs)
        total_input_tokens = sum(log.get("input_tokens", 0) for log in st.session_state.usage_logs)
        total_output_tokens = sum(log.get("output_tokens", 0) for log in st.session_state.usage_logs)
        total_cost = sum(log.get("est_cost_idr", 0) for log in st.session_state.usage_logs)
        
        col1.metric("Total Queries", total_queries)
        col2.metric("Total Input Tokens", total_input_tokens)
        col3.metric("Total Output Tokens", total_output_tokens)
        col4.metric("Total Cost (IDR)", f"Rp {total_cost:,.0f}")
