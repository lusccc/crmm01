from keybert import KeyBERT

doc = """
Closing of the transaction is expected to occur in the fourth quarter of 2012 following the completion of a marketing period in connection with the new third-party financing arranged by CD&R and is subject to certain customary conditions, including no occurrence of a Material Adverse Effect on the business since August 15, 2012, and obtaining regulatory approvals.In calculating the fair value of the reporting units or specific intangible assets, management relies on a number of factors, including operating results, business plans, economic projections, anticipated future cash flows, comparable transactions and other market data.
      """
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(doc, top_n=10)
print(keywords)