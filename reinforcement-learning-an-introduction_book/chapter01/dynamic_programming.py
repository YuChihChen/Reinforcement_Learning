
###########################################################################
#                                動態規劃                                  #
###########################################################################
"""
================================ 數學歸納法 ================================
定理：
    1. Y_0 成立。
    2. 假設 Y_1, Y_2, ..., Y_k 成立，如果能證明 Y_{k+1} 一定成立，
    3. 則 Y_n 成立，對所有的正整數 n


數學歸納法類型的問題：
    將一個問題分成多個小問題來解決，小問題又可以分成更小的子問題來解決，一直將問題切割下去
    ，一直切分到可以做得出來為止。
    如果：
        1. 切分的子問題，形式都和母問題一樣，只是複雜度不一樣，
        2. 母問題的解可以由子問題的解來組成，
    那由數學歸納得知，我們可得原始問題的解

    母問題 -> 拆分成子問題 --> 求解子問題 --> 合併子問題的解 --> 得到母問題的解 


如何判斷一個問題是數學歸納法類型？
    當題目有
        1. 變數：一個或多個數量級的非連續型變量，假設為 n，
        2. 目標：一個你想要計算的目標，其變數是 n，假設為 O(n)，
        3. 轉換：在 O(n-1) 已知的情況下(這時候先不要管它算不算的出來)，我就可以算出 O(n)
        4. 邊界：你可以解決邊界條件下的目標，例如 O(0)，
    這時候就是可以用數學歸納法來解這個問題


例子：
    - Factorial：n! = n * (n-1)!
    - Fibonacci number： F_n = F_{n-1} + F_{n-2}
    - 在曼哈頓，由起點到終點的走法數(不回頭)：N[m, n] = N[m-1, n] + N[m, n-1]
    - 有 2, 5, 7 元硬幣，用最少硬幣數量拼出 27 元幣值
    - 64*64 的白色棋盤如果一格塗黑，可否用 2*2 少一格的紙片將白色部位蓋滿
    - 數列排序問題，mergesort



=================== Iteration(迭代) or Recursion(遞迴)： ===================
概念：
    一個問題如果能用數學歸納法的形式寫出來，基本上演算法就已經得到了。再來只是決定用迭代
    方式還是遞迴方式來寫程式。但是解一個問題最難的也是如何找出這個數學歸納法的形式。
    底下都用 factorial 當做例子，求 n!=factorial(n)，

Iteration(迭代)： 函數裡面會有 loop
    # n! = 1 * 2 * 3 ... * n
    def factorial(n):
        output = 1
        for i in range(2, n+1):
            output = output * i
        return output 

Recursion(遞迴)：函數裡面會呼叫本身函數
    # n! = n * (n-1)!
    def factorial(n):
        if n == 1:
            return 1
        return n * factorial(n-1)



======================== 演算法架構： 分治法和動態規劃法 =======================
概念：
    分治法和動態規劃法只是將上述的數學歸納法問題分成兩個類型，子問題彼此間究竟獨立或不獨立
                    |-- 子問題彼此平行 -> 分治法
    數學歸納法問題  --
                    |-- 子問題互有重疊 -> 動態規劃法

分治法 (Divdide and Conquer)：
    子問題間彼此平行獨立，抽象來說，如果 A 分成 B1 和 B2 兩個子問題，則 B1 再分下去的所有
    子問題並不會和 B2 再分下去的所有子問題有重疊，不會有重複計算的問題
        
                    |-- C1
            -- B1 --|
            |       |-- C2
        A --|                   ====> {C1, C2} 交集 {C3, C4} = empty 
            |       |-- C3
            -- B2 --|
                    |-- C4
    
    例子：
        - Factorial：n! = n * (n-1)!
        - 64*64 的白色棋盤如果一格塗黑，可否用 2*2 少一格的紙片將白色部位蓋滿
        - 數列排序問題，mergesort

動態規劃法(Dynamic Programming，縮寫為 DP)：
    子問題間彼此並非平行獨立，抽象來說，如果 A 分成 B1 和 B2 兩個子問題，則 B1 再分下去
    的所有子問題會和 B2 再分下去的所有子問題有所重疊，存在重複計算的問題
        
                    |-- C1
            -- B1 --|
            |       |-- C2
        A --|                   ==> B1 和 B2 都重複計算了 C2
            |       |-- C2
            -- B2 --|
                    |-- C3
    
    例子：
        - Fibonacci number： F_n = F_{n-1} + F_{n-2}
        - 在曼哈頓，由起點到終點的走法數(不回頭)：N[m, n] = N[m-1, n] + N[m, n-1]
        - 有 2, 5, 7 元硬幣，用最少硬幣數量拼出 27 元幣值
    
    動態規劃法是 Richard E. Bellman 在 1953 年提出，當時他在美國軍方工作，取名該演算法
    叫做 Dynamic Programming 的原因是聽起來比較屌，老闆才會願意給研究資金。
    思路很簡單，空間換時間，算過的結果就存起來，就解決重複計算問題了。
    底下用 Fibonacci number 當做例子：F(n) = F(n-1) + F(n-2)
    
    問題： Get F(n) for all n <= 20
    
    DP 解法一： Iteration(迭代)
        # F0, F1, F2, F3 ..., F20
        def fibonacci(n):
            f_list = [1, 1]    # 將結果存在 f_list, F0 = 1 and F1 = 1
            for i in range(2, n+1):
                fi = f_list[i-1] + f_list[i-2]
                f_list.append(fi)
        return f_list

    DP 解法二：
        # F(n) = F(n-1) + F(n-2)
        N = 20
        f_list = [None] * (N+1)  # 將結果存在函數外面
        f_list[0] = 1
        f_list[1] = 1
        def fibonacci(n, f_list):
            if f_list[n] is None:
                f_list[n] = fibonacci(n-1, f_list) + fibonacci(n-2, f_list)
            return f_list[n]



========================== 作業一： 動態規劃法的練習 ==========================
Q1. 曼哈頓問題：
    假設在一個 M * N 的矩形區域內，你的出發點位於[0, 0]，終點位於[M, N]，
    每次只能走動 [1,0] or [0,1]，請問由 [0,0] 走到 [M,N] 有幾種走法？
    a. 使用 動態規劃法(迭代版本)
    b. 使用 動態規劃法(遞迴版本)

Q2. 硬幣組合問題：
    假設你有無限多個 2, 5, 7 元硬幣，請問要如何用最少硬幣數量拼出 N 元幣值？
    這裡只要求輸出最少要幾個就好，不需要輸出拼法。
    a. 使用 動態規劃法(迭代版本)
    b. 使用 動態規劃法(遞迴版本)
"""






# ========================= 防雷分隔線：作業參考解答 =========================






##########################################################################
#                           Q1: 曼哈頓問題                                #
##########################################################################
""" 
數學歸納法類型的問題
    1. 變數：二維變數 [m, n]，座標位置
    
    2. 目標：由 [0, 0] 走到 [m, n] 的走法數目，假設為 N[m, n]
    
    3. 轉換：如果我知道 N[m-1, n] 和 N[m, n-1]，那我就知道 N[m, n]
            i.e., N[m, n] = N[m-1, n] + N[m, n-1]
    
    4. 邊界：if m = 0 => N[m, n] = N[m, n-1]
            if n = 0 => N[m, n] = N[m-1, n]

數學歸納法的式子： N[m, n] = N[m-1, n] + N[m, n-1] 
"""

# 使用迭代版本的動態規劃法
def get_number_of_ways_by_iteration(m, n):
    answer = [[None] * (n+1) for _ in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 and j == 0:
                answer[i][j] = 1
            elif i == 0:
                answer[i][j] = answer[i][j-1]
            elif j == 0:
                answer[i][j] = answer[i-1][j]
            else:
                answer[i][j] = answer[i][j-1] + answer[i-1][j]
    return answer


# 使用遞迴版本的動態規劃法
def get_number_of_ways_by_recursion(m, n, answer):
    if answer[m][n] is not None:
        return answer[m][n]

    if m == 0 and n == 0:
        answer[m][n] = 1
    elif m == 0:
        answer[m][n] = get_number_of_ways_by_recursion(m, n-1, answer)
    elif n == 0:
        answer[m][n] = get_number_of_ways_by_recursion(m-1, n, answer)
    else:
        answer[m][n] = get_number_of_ways_by_recursion(m, n-1, answer)\
                     + get_number_of_ways_by_recursion(m-1, n, answer)
    return answer[m][n]


# 列印求出來的解
def print_answer_for_question_1(answer):
    number_of_digits = len(str(answer[-1][-1]))   # 得到要列印數字的最大位數
    m = len(answer)
    n = len(answer[0])
    print('Start') 
    for i in range(m):
        row_length = 0
        for j in range(n):
            str_ij = str(answer[i][j])
            num_of_space = number_of_digits - len(str_ij)
            str_to_print = ' ' * num_of_space + str_ij + ' '
            print(str_to_print, end='')
            row_length += len(str_to_print)
        print('')
    print(' ' * (row_length-2) + 'End')






##########################################################################
#                           Q2: 最少硬幣問題                               #
##########################################################################
"""
數學歸納法類型的問題
    1. 變數：單一變數 m，要組成的硬幣數值
    
    2. 目標：以最少的硬幣數量用2, 5, 7 三種硬幣組成 m，假設為 N[m]
    
    3. 轉換：如果我可以知道 N[m-2], N[m-5] 和 N[m-7] 的話，那我就知道 N[m]
            i.e., N[m] = min( N[m-7], N[m-5], N[m-2] )
    
    4. 邊界：m < 0 和 m = 1 的情況，沒有解，這時候我們令其 N[m] = 'inf'
            m = 0，N[0] = 0，因為我用 0 個硬幣就可以組出幣值 0
            

數學歸納法的式子： N[m] = min( N[m-7], N[m-5], N[m-2] ) 
"""

# 使用迭代版本的動態規劃法
def get_minimum_number_of_coins_iteration(m):
    answer = [None] * (m + 1)
    answer[0] = 0
    for i in range(1, m+1):
        Nim2 = answer[i-2] if i-2 >= 0 else float('inf')
        Nim5 = answer[i-5] if i-5 >= 0 else float('inf')
        Nim7 = answer[i-7] if i-7 >= 0 else float('inf')
        answer[i] = min(Nim2, Nim5, Nim7) + 1
    return answer
        

# 使用遞迴版本的動態規劃法
def get_minimum_number_of_coins_recursion(m, answer):
    if answer[m] is not None:
        return answer[m]
    if m < 0:
        return float('inf')

    if m == 0:
        answer[m] = 0
    else:
        a = get_minimum_number_of_coins_recursion(m-2, answer)
        b = get_minimum_number_of_coins_recursion(m-5, answer)
        c = get_minimum_number_of_coins_recursion(m-7, answer)
        answer[m] = min(a, b, c) + 1
    return answer[m]
     





##########################################################################
#                           執行程式並顯示結果                              #
##########################################################################
if __name__ == '__main__':

    print('==================== Q1: 曼哈頓問題 ====================')
    m, n = 5, 7
    # --- 迭代版本 ---
    print('------ Output: 動態規劃(迭代) ------')
    answer_q1_iter = get_number_of_ways_by_iteration(m, n)
    print_answer_for_question_1(answer_q1_iter)
    print('')
    # --- 遞迴版本 ---
    print('------ Output: 動態規劃(遞迴) ------')
    answer_q1_recu = [[None] * (n+1) for _ in range(m+1)]
    get_number_of_ways_by_recursion(m, n, answer_q1_recu)
    print_answer_for_question_1(answer_q1_recu)


    
    print('')
    print('==================== Q2: 最少硬幣問題 ====================')
    m = 27
    
    # --- 迭代版本 ---
    answer_q2_iter = get_minimum_number_of_coins_iteration(m)
    print('------ Output: 動態規劃(迭代) ------')
    print(f'用 2 , 5, 7 元的硬幣組成 {m} 元最少需要 {answer_q2_iter[m]} 個硬幣')
    print('')
    
    # --- 遞迴版本 ---
    answer_q2_recu = [None] * (m + 1)
    get_minimum_number_of_coins_recursion(m, answer_q2_recu)
    print('------ Output: 動態規劃(遞迴) ------')
    print(f'用 2 , 5, 7 元的硬幣組成 {m} 元最少需要 {answer_q2_recu[m]} 個硬幣')
