SELECT b.title, c.category_name, b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE c.category_name IN ('Travel', 'Mystery', 'Historical Fiction');