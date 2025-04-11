import numpy as np
import time
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingBenchmark:
    def __init__(self, test_texts: List[str]):
        """
        Initialize benchmark with test texts
        Args:
            test_texts: List of representative text samples from dataset
        """
        self.test_texts = test_texts
        self.models = {
            # Lightweight
            "all-MiniLM-L6-v2": {"dim": 384, "speed": "fast"},
            "BAAI/bge-small-en-v1.5": {"dim": 384, "speed": "fast"},

            # Balanced
            "all-mpnet-base-v2": {"dim": 768, "speed": "medium"},
            "BAAI/bge-base-en-v1.5": {"dim": 768, "speed": "medium"},

            # Powerful
            "BAAI/bge-large-en-v1.5": {"dim": 1024, "speed": "slow"},
            "intfloat/e5-large-v2": {"dim": 1024, "speed": "slow"}
        }

    def _warmup(self, model):
        """Warmup GPU"""
        warmup_text = ["warmup"] * 8
        model.encode(warmup_text, batch_size=8)

    def _time_inference(self, model_name: str, batch_size: int = 32) -> Dict:
        """Benchmark single model"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
        self._warmup(model)

        # Speed test
        start = time.time()
        embeddings = model.encode(
            self.test_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        duration = time.time() - start

        # Quality test (self-similarity)
        sample = embeddings[:5]  # Check first 5 embeddings
        similarities = cosine_similarity(sample)
        np.fill_diagonal(similarities, 0)  # Remove self-similarity
        avg_similarity = np.mean(similarities)

        return {
            "model": model_name,
            "dim": self.models[model_name]["dim"],
            "speed_sec": round(duration, 2),
            "speed_per_doc": round(duration/len(self.test_texts), 4),
            "avg_similarity": round(avg_similarity, 4),
            "embeddings_shape": embeddings.shape,
            "batch_size": batch_size,
            "device": device
        }

    def run_benchmark(self, batch_sizes: List[int] = [32, 64]) -> List[Dict]:
        """Compare all models"""
        results = []

        for model_name in self.models:
            for batch_size in batch_sizes:
                try:
                    print(f"Testing {model_name} with batch_size={batch_size}...")
                    result = self._time_inference(model_name, batch_size)
                    results.append(result)
                except Exception as e:
                    print(f"Failed {model_name} at batch_size={batch_size}: {str(e)}")

        # Sort by speed then quality
        return sorted(results, key=lambda x: (x["speed_sec"], -x["avg_similarity"]))

    def print_results(self, results: List[Dict]):
         """Pretty print results"""
         print("\nBenchmark Results:")
         print(f"{'Model':<30} {'Dim':<6} {'Speed(s)':<8} {'Speed/doc':<10} {'Similarity':<10} {'Batch':<6} {'Device':<6}")
         print("-"*80)
         for r in results:
             print(f"{r['model']:<30} {r['dim']:<6} {r['speed_sec']:<8} {r['speed_per_doc']:<10} {r['avg_similarity']:<10} {r['batch_size']:<6} {r['device']:<6}")

if __name__ == "__main__":
    # Example usage
    test_texts = [
        "§31¦ December 2016\n(AMD'000)\nCaptions 3 4 5 7 9 10 11 12 13 14\nComparative interim period of prior financial year\nBalance at the beginning of the current period, as at 1 January 2015 30,000,000 0 4,595,192 576,817 3,888,683 13,683,875 0 52,744,567 1,548,304 54,292,871\n9.1.Total impact of changes in accounting policy and the correction of fundamental 0\nerrors\n10. Restated balance 30,000,000 0 4,595,192 576,817 3,888,683 13,683,875 0 52,744,567 1,548,304 54,292,871\n12. Transactions with shareholders (owners) with respect to shares 0 -\n(shareholdings)\n12.1. Investments in share capital and increase of share capital 0 -\n12.2. Decrease in the share capital as a result of purchased back shares 0 -\n13.Comprehensive Income (149,539) (319,273) (41,689) (510,501) 1,668 (508,833)\n14.Dividends distributed 0 ( 71,910) ( 71,910)\n15. Other increase/(decrease) of equity components, including 0 -\n15.1.Other increase/(decrease) of equity components 0 -\n16.Internal movements, including 0 0 0 - (579,291) 579,291 0 -\n16.1.Allotment to general reserve 0 -\n16.2. Cover of loss from general reserve 0 -\n16.3.Cover of share discount 0 -\n16.4.Decrease in value of property, plant and equipment and other intangible (579,291) 579,291 0 -\nassets caused by revaluation\n16.5.Internal movements of other equity components 0 -\nBalance at the end of the current period, as at 01 January 2016 0 -\n17.Balance at the beginning of the current period, as at 1 January 2016 30,000,000 0 4,595,192 427,278 2,990,119 14,221,477 0 52,234,066 1,478,062 53,712,128\nlatipaC\nerahS\nevreser\nlareneG\nstessa\nelas-rof-elbaliava\nfo\nnoitaulaveR\n)ssol(/sgninrae\ndeniateR\nstnemyap\ndesab\nerahS\nnoitapicitrap\ngnillortnoc-non\nlitnu\nlatoT\nstseretni\ngnillortnoc-noN\nlatoT\nACBA-CREDIT AGRICOLE BANK CJSC\nEquity components\nmuimerp\nerahS\nstessa\ntnerruc-non\nfo\nnoitaulaveR\nInterim Financial Statements\nStatement of Changes in Equity\nAddress: 82-84 Aram Street, Yerevan, Armenia\nInterim period of current financial year\n1. Balance at the beginning of the current period, as at 1 January 2016 30,000,000 0 4,595,192 427,278 2,990,119 14,221,477 0 52,234,066 1,478,062 53,712,128\n9.1. Total impact of changes in accounting policy and material errors 0\n10.Restated balance 30,000,000 0 4,595,192 427,278 2,990,119 14,221,477 52,234,066 1,478,062 53,712,128\n12.Transactions with shareholders (owners) with respect to shares 0 -\n(shareholdings) 0\n12.1. Investments in share capital and increase of share capital 0 -\n12.2. Decrease in share capital as a result of purchased back shares 0 -\n13.Comprehensive Income 1,023,180 (73,197) 2,083,274 3,033,257 84,599 3,117,856\n14.Dividends distributed 0 -\n15.Other increase/(decrease) of equity components 0 -\n15.1.Increase or decrease in derivatives classified as equity components 0 -\n16.Internal movements 0 0 0 - (8,383) 8,383 0 -\n16.1. Allotment to general reserve 0 -\n16.2.Cover of loss from general reserve 0 -\n16.3.Cover of share discount 0 -\n16.4. Decrease in value of property, plant and equipment and other intangible (8,383) 8,383 0 -\nassets caused by revaluation\n16.5. Internal movements of other equity components 0 -\nNon-controlling interests 0 -\nBalance at the end of the current period, as at 01 January 2017 30,000,000 0 4,595,192 1,450,458 2,908,539 16,313,134 0 55,267,323 1,562,661 56,829,984\nGeneral Executive Director H. Andreasyan\nA. Hakobyan\nChief Accountant\n",
        "\"Byblos Bank Armenia\" CJSC\nReport on prudential ratios\nas at 30 September 2021\nin '000 Drams\nActual value of Threshold, set by Breaches during the\nthe ratio the CBA quarter\nMinimum share capital 26,249,100 50,000 No breaches\nMinimum total equity 34,506,513 30,000,000 No breaches\nN11 - Minimum ratio of total equity to risk-weighted assets 34.43% 9.00% No breaches\nN12 - Minimum ratio of total equity to risk-weighted assets 42.13% 12.00% No breaches\nN21 - Minimum ratio of high-liquid assets to total assets 44.82% 15.00% No breaches\nN22 - Minimum ratio of high-liquid assets to demand liabilities 303.71% 60.00% No breaches\nN211 – General liquidity ratio calculated for 1st basket currencies 39.97% 4.00% No breaches\nN212– General liquidity ratio calculated for USD, EUR and a 2nd basket\ncurrency Ï/ã¿ 4.00% No breaches\nN221 – Current liquidity ratio calculated for 1st basket currencies 256.77% 10.00% No breaches\nN222– Current liquidity ratio calculated for USD, EUR and a 2nd basket\ncurrency Ï/ã¿ 10.00% No breaches\nN23– Liquidity coverage ratio (LCR) 231.02% 100.00% No breaches\nN23 (AMD) – Liquidity coverage ratio (LCR) 343.27% 80.00% No breaches\nN23 (1st basket significant currencies) – Liquidity coverage ratio (LCR) 156.89% 80.00% No breaches\nN24– Net stable funding ratio (NSFR) 194.68% 100.00% No breaches\nN24 (AMD) – Net stable funding ratio (NSFR) 202.08% 80% No breaches\nN24 (1st basket significant currencies) – Net stable funding ratio (NSFR) 187.00% 80.00% No breaches\nN31 - Maximum exposure per one borrower 15.30% 20.00% No breaches\nN32 - Maximum exposure per all large borrowers 75.06% 500.00% No breaches\nN41 - Maximum exposure per one bank-related party 4.42% 5.00% No breaches\nN42 - Maximum exposure per all bank-related parties 10.42% 20.00% No breaches\nRequired reservation in the Central Bank of Armenia No breaches\nin AMD - 4.00% No breaches\nin USD - 18.00% No breaches\nin EUR - 18.00% No breaches\nin other currencies - 18.00% No breaches\nMaximum ratio of long position in each foreign currency and total equity\nin USD 0.15% 7.00% No breaches\nin EUR 0.00% 7.00% No breaches\nin RUB 0.02% 7.00% No breaches\nin other currencies 0.01% 7.00% No breaches\nMaximum ratio of aggregate long positions in all foreign currencies and total\nequity 0.18% 10.00% No breaches\n",
        "Interim Financial Statements\nIncome statement\n§31¦ December 2016\nACBA-CREDIT AGRICOLE BANK CJSC\nAddress: 82-84 Aram Street, Yerevan, Armenia\n. (AMD'000)\n01.10.16- 01.01.16-31.12.16 01.10.15-31.12.15 01.01.15-31.12.15\n31.12.16\nName\nInterest income 7,771,657 31,567,159 7,999,410 32,226,144\nInterest expenses (3,445,699) ( 13,744,431) ( 3,735,654) (15,192,877)\nNet interest income 4,325,958 17,822,728 4,263,756 1 7,033,267\nFee and commission income 1,029,204 3,561,364 665,445 3,265,417\nFee and commission expenses (598,797) ( 1,820,589) ( 139,314) (1,252,189)\nNet fee and commission income 4 30,407 1,740,775 526,131 2 ,013,228\n- - -\nDividend income - 3,765 1,246 5,623\nNet income from trade operations 470,776 1,273,326 659,513 1,529,861\nOther operating income 1,078,586 3,214,098 21,670 1,572,966\nOperating income 6,305,727 24,054,692 5,472,316 2 2,154,945\nImpairment losses (1,284,358) ( 6,802,077) ( 367,137) (8,638,025)\nGeneral administrative expenses (3,451,455) ( 10,737,583) ( 3,027,348) (9,707,095)\nOther operating expenses (1,326,819) ( 3,386,780) ( 961,628) (3,448,931)\nNet profit/(loss) from investments in controled entities 8,142 ( 13,303) ( 11,205) (3,149)\nProfit/ (loss) before taxes 251,237 3,114,949 1,104,998 3 57,745\nIncome tax charge (134,541) ( 947,076) ( 395,655) (397,766)\nNet Profit/ (loss) for the period, including: 116,696 2,167,873 709,342 (40,021)\nEquity holders of parent entity 9 2,006 2,083,274 569,343 (41,689)\nNon-controlling interests 2 4,691 84,599 139,999 1,668\nOther comprehensive income - - - -\nCurrency translation differences of the financial statements with\na foreign operation - - - -\nRevaluation of available -for-sale assets (166,760) 1,206,057 (90,181) (586,015)\nCash Flow hedges - - - -\nRevaluation of non-current assets - - - -\nIncome tax on comprehensive income (256,074) (256,074) 117,203 117,203\nOther comprehensive income after tax\nTotal comprehensive income, including: (306,138) 3,117,856 736,364 (508,833)\n- - - -\nEquity holders of parent entity (330,828) 3,033,257 596,365 (510,501)\nNon-controlling interests 2 4,691 84,599 139,999 1,668\nGeneral Executive Director H. Andreasyan\nChief Accountant A. Hakobyan\n",
        "Statement of Financial Position\n§31¦ December 2016\nACBA-CREDIT AGRICOLE BANK CJSC\nAddress: 82-84 Aram Street, Yerevan, Armenia\n(AMD'000)\nNotes As at the end of the current period As at the end of prior financial\nName\n31/12/16 year 31/12/15\n1 Assets\n1.1\nCash and balances with the Central Bank of Armenia 13 51,333,568 42,336,742\n1.2 Precious metals bullions\nPlacement with banks and other financial institutions\n1.3\n14 12,662,729 13,016,350\n1.4 Financial assets held for trading 15 24,974 2,638,083\n1.5 Loans and advances to customers 16 165,543,359 168,358,966\n1.6 Receivables from finance leases 16 9,394,594 9,607,216\n1.7 Financial assets available-for- sale 17 26,275,179 17,350,075\n1.8 Investments held to maturity 18\n1.9 Investments in share capital of controlled entities 19 200,664 243,032\n1.10 Not current assets held for disposal\n1.11 Property, plant and equipment and other intangible assets 20 15,449,764 15,278,358\n1.12 Defferred tax assets 12\n1.13 Other assets 22 2,939,733 4,024,416\nTotal assets 283,824,564 272,853,238\n2 Liabilities\nDeposits and balances from banks and other financial\n2.1\ninstitutions 23 87,765,070 84,146,562\n2.2 Current accounts and deposits from customers 24 136,192,416 131,487,594\n2.3 Securities issued by the Bank 25\n2.4 Liabilities held for trading 26 1,891 51,658\n2.5 Amounts payable 27 57,233 41,238\n2.6 Current tax liabilities\n2.7 Deferred tax liabilities 12 803,597 931,511\n2.8 Provisions 34\n2.9 Other liabilities 29 2,174,373 2,482,547\nTotal liabilities 226,994,580 219,141,110\n3 Equity\n3.1 Share capital 30 30,000,000 30,000,000\n3.2 Share premium 31\n3.3 Reserves: 8,954,189 8,012,589\n3.3.1 General reserve 4,595,192 4,595,192\n3.3.2 Revaluation reserve 32 4,358,997 3,417,397\n3.4 Retained earnings/(loss) 33 16,313,134 14,221,477\n3.5 Other equity components\nEquity holders of parent entity 55,267,323 52,234,066\nNon-controlling interests 1,562,661 1,478,062\nTotal equity 56,829,984 53,712,128\nTotal equity and liabilities 283,824,564 272,853,238\nGeneral Executive Director\nH. Andreasyan\n",
    ]

    benchmark = EmbeddingBenchmark(test_texts)
    results = benchmark.run_benchmark(batch_sizes=[32, 64, 128])
    benchmark.print_results(results)