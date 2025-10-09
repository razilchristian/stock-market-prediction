from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 102, 204)
        self.cell(0, 10, "📅 8-Month Financial & Studio Launch Plan (May - December 2025)", ln=True, align="C")
        self.ln(5)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(255, 255, 255)
        self.set_fill_color(0, 102, 204)
        self.cell(0, 10, f"  {title}", ln=True, fill=True)
        self.ln(3)

    def section_body(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 8, text)
        self.ln()

pdf = PDF()
pdf.add_page()

pdf.section_title("1. Income & Expense Overview")
pdf.section_body(
    "• Monthly Income: ₹42,000\n"
    "• Total EMI (May–July): ₹25,045/month\n"
    "• EMI (Aug–Sep): ₹17,995/month\n"
    "• EMI (Oct–Dec): ₹13,331/month\n"
    "• Recommended Monthly Expenses: ₹10,000–₹12,000\n"
    "• Savings Plan: Begin saving from May by controlling expenses"
)

pdf.section_title("2. EMI Timeline")
pdf.section_body(
    "• May–June: Pay all 6 EMIs (₹25,045)\n"
    "• July: Two EMIs (₹3,550 & ₹3,500) end\n"
    "• August–September: Pay ₹17,995\n"
    "• October: ₹4,664 EMI ends\n"
    "• October–December: Remaining EMIs total ₹13,331/month"
)

pdf.section_title("3. Monthly Savings Targets")
pdf.section_body(
    "• May: ₹5,000–₹7,000\n"
    "• June: ₹5,000–₹7,000\n"
    "• July: ₹10,000\n"
    "• August: ₹10,000–₹12,000\n"
    "• September: ₹12,000\n"
    "• October: ₹15,000\n"
    "• November: ₹15,000\n"
    "• December: ₹15,000"
)

pdf.section_title("4. Studio Launch Plan (January 2026)")
pdf.section_body(
    "• Type: Maternity, Baby Shoots, Podcast, Content Creation\n"
    "• Location: Ahmedabad (rented space)\n\n"
    "📸 Estimated Budget:\n"
    "• Rent & Deposit: ₹30,000–₹40,000\n"
    "• Equipment (Camera, Lights): ₹40,000–₹50,000\n"
    "• Backdrops & Props: ₹20,000\n"
    "• Podcast & Branding: ₹15,000\n"
    "• TOTAL: ₹1.2–₹1.5 Lakh"
)

pdf.section_title("5. Final Tips & Action Plan")
pdf.section_body(
    "• Use a separate savings account to avoid spending studio funds\n"
    "• Track expenses closely and reduce non-essentials\n"
    "• Offer weekend freelance shoots to increase income\n"
    "• Begin branding (name, logo, Instagram) by October"
)

pdf.output("Financial_Studio_Launch_Plan.pdf")
