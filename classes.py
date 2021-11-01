class BankAccount:

    balance = 0

    def withdraw(self, amount: int):
        self.balance = self.balance - amount

    def deposit(self, amount: int):
        self.balance = self.balance + amount

    def __init__(self, startingBalance: int):
        self.balance = startingBalance


bank_account_1 = BankAccount(100)
bank_account_2 = BankAccount(50)

bank_account_1.withdraw(50)
bank_account_2.withdraw(50)

print(bank_account_1.balance) # 50
print(bank_account_2.balance) # 0
