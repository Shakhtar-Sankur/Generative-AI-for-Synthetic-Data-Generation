SELECT
    disease_label,
    COUNT(*) as image_count,
    AVG(LENGTH(image_data)) as avg_image_size
FROM `your-project.dataset.synthetic_images`
GROUP BY disease_label
ORDER BY image_count DESC;