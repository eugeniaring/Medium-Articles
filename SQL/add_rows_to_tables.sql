INSERT INTO Customers (customer_id, name, email, region, join_date) VALUES
(1, 'Alice Smith', 'alice.smith@example.com', 'Europe', '2023-01-15'),
(2, 'Bob Johnson', 'bob.johnson@example.com', 'North America', '2023-02-20'),
(3, 'Charlie Brown', 'charlie.brown@example.com', 'Europe', '2023-03-10'),
(4, 'Daisy Ridley', 'daisy.ridley@example.com', 'Asia', '2023-05-05'),
(5, 'Eve White', 'eve.white@example.com', 'Europe', '2023-06-25'),
(6, 'Frank Black', 'frank.black@example.com', 'North America', '2024-01-15'),
(7, 'Grace Green', 'grace.green@example.com', 'Asia', '2024-02-10'),
(8, 'Henry Gray', 'henry.gray@example.com', 'Europe', '2024-03-05'),
(9, 'Ivy Blue', 'ivy.blue@example.com', 'Asia', '2024-04-20'),
(10, 'Jack White', 'jack.white@example.com', 'North America', '2024-05-15');

INSERT INTO Categories (category_id, name) VALUES
(1, 'Clothing'),
(2, 'Shoes'),
(3, 'Accessories'),
(4, 'Sportswear');

INSERT INTO Products (product_id, name, category_id, price, stock) VALUES
(1, 'Jeans', 1, 49.99, 100),
(2, 'T-Shirt', 1, 19.99, 200),
(3, 'Running Shoes', 2, 89.99, 50),
(4, 'Sneakers', 2, 69.99, 120),
(5, 'Wristwatch', 3, 129.99, 30),
(6, 'Sunglasses', 3, 79.99, 75),
(7, 'Tracksuit', 4, 99.99, 60),
(8, 'Jacket', 1, 149.99, 40),
(9, 'Sandals', 2, 39.99, 150),
(10, 'Hat', 3, 24.99, 100);

INSERT INTO Orders (order_id, customer_id, order_date) VALUES
(1, 1, '2023-03-15'),
(2, 2, '2023-04-10'),
(3, 3, '2023-05-05'),
(4, 4, '2023-06-20'),
(5, 5, '2023-07-10'),
(6, 6, '2024-01-15'),
(7, 7, '2024-02-15'),
(8, 8, '2024-03-05'),
(9, 9, '2024-04-10'),
(10, 10, '2024-05-25');

INSERT INTO OrderDetails (order_detail_id, order_id, product_id, quantity) VALUES
(1, 1, 1, 2),
(2, 1, 2, 3),
(3, 2, 3, 1),
(4, 2, 4, 2),
(5, 3, 5, 1),
(6, 3, 6, 2),
(7, 4, 7, 1),
(8, 4, 8, 1),
(9, 5, 9, 4),
(10, 5, 10, 1),
(11, 6, 1, 1),
(12, 6, 3, 1),
(13, 7, 2, 2),
(14, 7, 6, 1),
(15, 8, 5, 1),
(16, 8, 4, 1),
(17, 9, 7, 3),
(18, 9, 8, 2),
(19, 10, 9, 1),
(20, 10, 10, 2);
