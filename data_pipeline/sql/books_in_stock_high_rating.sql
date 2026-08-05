SELECT title, price_gbp, rating
        FROM books
        WHERE in_stock = 1 AND rating >= 4;