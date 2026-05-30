result = DeepFace.represent(
    img_path=rostro_recortado,
    model_name='Facenet',
    detector_backend='skip',
    enforce_detection=False
)

query_embedding = np.array(result[0]['embedding'])

cos_sim = np.dot(query_embedding, ref_embedding) / (
    np.linalg.norm(query_embedding) *
    np.linalg.norm(ref_embedding)
)

distancia = 1 - cos_sim