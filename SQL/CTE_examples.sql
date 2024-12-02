-- Example 1: Simple CTE

WITH NumberCustomerByYear AS (
   SELECT STRFTIME('%Y',c.join_date) AS Year, count(*) AS NumberCustomers
   FROM Customers c
   GROUP BY STRFTIME('%Y',c.join_date)

)

SELECT *
FROM NumberCustomerByYear
ORDER BY Year DESC;

-- Example 2: Simplify a Complex Query

WITH PopularProducts AS (
 SELECT
   p.name AS product_name,
   SUM(od.quantity) AS total_quantity_sold
 FROM Products p 
 LEFT JOIN OrderDetails od ON p.product_id = od.product_id
 GROUP BY p.name
 HAVING SUM(od.quantity)>3
)

SELECT *
FROM PopularProducts
ORDER BY total_quantity_sold DESC;

-- Example 3: Use Multiple CTEs in a Query

WITH MonthlyOrders AS (
    SELECT 
        STRFTIME('%Y',order_date) AS order_year,
        CAST(STRFTIME('%m',order_date) AS INTEGER) AS order_month,
        COUNT(order_id) AS total_orders
    FROM Orders
    GROUP BY STRFTIME('%Y',order_date), STRFTIME('%m',order_date)
),
MonthlyComparison AS (
    SELECT 
        mo1.order_year,
        mo1.order_month,
        mo1.total_orders AS current_month_orders,
        COALESCE(mo2.total_orders, 0) AS previous_month_orders,
        mo1.total_orders - COALESCE(mo2.total_orders, 0) AS order_difference
    FROM MonthlyOrders mo1
    LEFT JOIN MonthlyOrders mo2 
        ON (mo1.order_year = mo2.order_year AND mo1.order_month = mo2.order_month + 1)
         OR (mo1.order_year = mo2.order_year+1 AND mo1.order_month=1 AND mo2.order_month=12)
)
SELECT *
FROM MonthlyComparison
ORDER BY order_year DESC, order_month DESC;
