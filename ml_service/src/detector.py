import torch
import torch.nn as nn
import pickle
import os
import pandas as pd
from src.data_loader import DataLoader
from src.models.bilstm_model import BiLSTMEncoder, Vocab

class ICD10Detector:
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "full_raw_codes.csv")
            
        self.data_path = data_path
        loader = DataLoader(data_path)
        self.df = loader.load_data()
        
        self.codes = self.df['code'].tolist()
        self.descriptions = self.df['description'].tolist()
        
        # BiLSTM Contrastive Encoder Setup
        self.model_dir = os.path.join(os.path.dirname(__file__), "models", "weights")
        self.vocab_path = os.path.join(self.model_dir, "model_metadata.pkl")
        self.weight_path = os.path.join(self.model_dir, "bilstm_icd10.pth")

        print("Loading BiLSTM Encoder and Vocabulary...")
        with open(self.vocab_path, "rb") as f:
            metadata = pickle.load(f)
            self.vocab = metadata['vocab']
            
        self.model = BiLSTMEncoder(vocab_size=len(self.vocab))
        self.model.load_state_dict(torch.load(self.weight_path, map_location=torch.device("cpu")))
        self.model.eval()
        
        print(f"Pre-computing embeddings for {len(self.descriptions)} codes...")
        with torch.no_grad():
            batch_size = 512
            all_embeddings = []
            for i in range(0, len(self.descriptions), batch_size):
                batch_texts = self.descriptions[i:i+batch_size]
                encoded = [self.vocab.encode(t) for t in batch_texts]
                tensor_input = torch.tensor(encoded)
                embeddings = self.model(tensor_input)
                all_embeddings.append(embeddings)
            self.description_embeddings = torch.cat(all_embeddings, dim=0)

        self.medical_synonyms = {
            "heart attack": "myocardial infarction",
            "flu": "influenza",
            "broken": "fracture",
            "high blood pressure": "hypertension",
            "cold": "nasopharyngitis",
            "sore throat": "pharyngitis"
        }

    def expand_with_synonyms(self, text):
        text = text.lower()
        expanded_terms = [text]
        for key, val in self.medical_synonyms.items():
            if key in text:
                expanded_terms.append(text.replace(key, val))
        return list(set(expanded_terms))

    def detect(self, user_input, top_k=3):
        expanded_queries = self.expand_with_synonyms(user_input)
        all_results_dict = {}
        # Filter out common clinical 'filler' words to focus on diagnostic terms
        clinical_stopwords = {'patient', 'presents', 'with', 'caused', 'exposure', 'resulting', 'acute', 'symptoms', 'illness', 'unwellness', 'sickness', 'presented', 'intoxication'}
        pathogen_keywords = {'clostridium', 'perfringens', 'staphylococcus', 'streptococcus', 'salmonella', 'influenza', 'cholera', 'typhoid', 'herpes', 'pneumonia'}
        
        with torch.no_grad():
            for query in expanded_queries:
                # Identify diagnostic terms (words not in stopwords and > 4 chars)
                words = query.lower().split()
                diag_terms = [t for t in words if t not in clinical_stopwords and len(t) > 3]
                
                # 1. CANDIDATE GENERATION (Precision Tiers)
                candidate_indices = []
                for idx, desc in enumerate(self.descriptions):
                    desc_l = desc.lower()
                    # Base match score
                    match_score = sum(3 if term in desc_l else 0 for term in diag_terms)
                    # Pathogen Priority Match
                    match_score += sum(10 if term in pathogen_keywords and term in desc_l else 0 for term in diag_terms)
                    
                    if match_score >= 3:
                        candidate_indices.append((idx, match_score))
                
                if not candidate_indices:
                    v_indices = list(range(len(self.codes)))
                    cand_weights = {i: 0 for i in v_indices}
                else:
                    candidate_indices = sorted(candidate_indices, key=lambda x: x[1], reverse=True)[:300]
                    v_indices = [c[0] for c in candidate_indices]
                    cand_weights = {c[0]: c[1] for c in candidate_indices}

                # 2. NEURAL RANKING
                query_indices = torch.tensor([self.vocab.encode(query)])
                query_embedding = self.model(query_indices)
                
                cand_embeddings = self.description_embeddings[v_indices]
                sim_scores = torch.mm(query_embedding, cand_embeddings.T)[0]
                
                tk = min(top_k * 5, len(v_indices))
                top_v = torch.topk(sim_scores, k=tk)
                
                for score, v_idx in zip(top_v[0], top_v[1]):
                    idx = v_indices[v_idx]
                    code = self.codes[idx]
                    m_score = cand_weights.get(idx, 0)
                    n_sim = float(score)
                    
                    if code not in all_results_dict or m_score > all_results_dict[code]['match_score']:
                        all_results_dict[code] = {
                            "code": code,
                            "description": self.descriptions[idx],
                            "match_score": m_score,
                            "neural_similarity": n_sim,
                            "score": min(1.0, (m_score / 20.0) + (n_sim * 0.4)),
                            "query_used": query
                        }
        
        # FINAL SORT: MATCH SCORE (Primary) -> NEURAL SIMILARITY (Secondary)
        all_results = list(all_results_dict.values())
        all_results.sort(key=lambda x: (x['match_score'], x['neural_similarity']), reverse=True)
        
        final_list = all_results[:top_k]
        for res in final_list:
            res['reasoning'] = self.generate_reasoning(user_input, res)
            
        return final_list
        
        # FINAL SORT: MATCH SCORE (Primary) -> NEURAL SIMILARITY (Secondary)
        all_results.sort(key=lambda x: (x['match_score'], x['neural_similarity']), reverse=True)
        final_list = all_results[:top_k]
        for res in final_list:
            res['reasoning'] = self.generate_reasoning(user_input, res)
            
        return final_list

    def generate_reasoning(self, user_input, result):
        reasoning = f"Matched '{user_input}' to '{result['description']}' through clinical term alignment ({result['match_score']} keywords). Total confidence {int(result['score']*100)}%."
        if "." in result['code']:
            parent = result['code'].split('.')[0]
            reasoning += f" Hierarchical match found in category {parent}."
        return reasoning
