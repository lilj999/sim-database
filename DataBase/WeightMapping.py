import math

class WeightMapping:
    def __init__(self):
        # 初始化权重映射
        self.role_weights = {
            "President": 5,
            "Manager": 3,
            "Engineer": 2,
            "IT": 2,
            "Statistician": 2,
            "Rest": 0.5,
        }

        self.function_unit_weights = {
            "ResearchAndEngineering": 5,
            "SalesAndMarketing": 4,
            "Manufacturing": 3,
            "Administration": 2,
            "PurchasingAndContracts": 1,
            "Finance": 1,
        }

        self.activity_weights = {
            "on": self.calculate_on_weight,
            "off": self.calculate_off_weight,
            "search": self.calculate_search_weight,
        }

        self.file_type_weights = {
            ".zip": 3,
            ".doc": 2.5,
            ".pdf": 2,
            ".jpg": 1.5,
            ".txt": 1,
        }

        self.time_feature_weights = {
            "Work": 1,
            "Off_Work": 10,
        }

        self.content_weights = {
            "confidential": 10,
            "proprietary": 10,
            "strategy": 10,
            "finance": 7,
            "contract": 7,
            "client": 7,
            "project": 7,
            "design": 7,
            "research": 5,
            "patent": 5,
        }

    def calculate_on_weight(self, n):
        """
        根据活动次数计算 'on' 的权重。
        """
        if n <= 10:
            return 2 + 0.5 * n  # 线性增长
        elif n <= 20:
            return 10 + 0.7 * (n - 10)  # 稍快的增长
        else:
            return 17 + 0.05 * (n - 20)**2  # 明显增长

    def calculate_off_weight(self, n):
        """
        根据活动次数计算 'off' 的权重。
        """
        if n <= 10:
            return 1 + 0.4 * n  # 线性增长
        elif n <= 20:
            return 5 + 0.5 * (n - 10)  # 稍快的增长
        else:
            return 10 + 0.03 * (n - 20)**2  # 明显增长

    def calculate_search_weight(self, n):
        """
        根据活动次数计算 'search' 的权重。
        """
        if n <= 10:
            return 2 + 0.6 * n  # 线性增长
        elif n <= 20:
            return 12 + 0.8 * (n - 10)  # 稍快的增长
        else:
            return 20 + 0.1 * (n - 20)**2  # 明显增长

    def calculate_total_weight(self, row_weights):
        """
        计算一行数据的所有权重总和。

        参数:
            row_weights (dict): 包含关键词和权重的字典。

        返回:
            float: 权重总和。
        """
        return sum(row_weights.values())
    
    def get_time_feature_weight(self, time_feature):
        """
        获取时间特征的权重。

        参数:
            time_feature (str): 时间特征 ('Work' 或 'Off_Work')。

        返回:
            float: 对应的权重值。
        """
        return self.time_feature_weights.get(time_feature, 0)

    def get_role_weight(self, role, is_resigned=False):
        # 模糊匹配角色
        for key in self.role_weights.keys():
            if key.lower() in role.lower():
                weight = self.role_weights[key]
                return weight * 5 if is_resigned else weight
        return 0

    def get_function_unit_weight(self, function_unit):
        # 模糊匹配功能单位
        for key in self.function_unit_weights.keys():
            if key.lower() in function_unit.lower():
                return self.function_unit_weights[key]
        return 0

    def get_activity_weight(self, activity, n):
        """
        根据活动类型和次数计算权重。

        参数:
            activity (str): 活动类型 ('on', 'off', 'search')。
            n (int): 活动次数。

        返回:
            float: 计算得到的权重。
        """
        # 直接使用归类后的 activity
        if activity in self.activity_weights:
            return self.activity_weights[activity](n)
        return 0


    def extract_keywords_weights(self, dataframe):
        """
        接收数据处理函数的输出，并提取每个关键词及对应的权重。

        参数:
            dataframe (DataFrame): 数据处理函数的输出（包含 activity, role, function_unit, content, Time_Feature 等列）。

        返回:
            list: 包含每行提取的关键词及对应权重的列表（按需求格式化为 'Keyword': Weight）。
        """
        extracted_keywords = []

        for _, row in dataframe.iterrows():
            row_weights = {}

            # 提取角色权重
            role = row.get("role", "")
            role_weight = self.get_role_weight(role, row.get("is_resigned", False))
            if role and role_weight > 0:
                row_weights[role] = role_weight

            # 提取功能单位权重
            function_unit = row.get("function_unit", "")
            function_unit_weight = self.get_function_unit_weight(function_unit)
            if function_unit and function_unit_weight > 0:
                row_weights[function_unit] = function_unit_weight

            # 提取活动权重
            activity = row.get("activity", "search").lower()

            # 统一处理 activity
            if "connect" in activity or "logon" in activity:
                activity = "on"
            elif "disconnect" in activity or "logoff" in activity:
                activity = "off"

            daily_operation_count = row.get("daily_operation_count", 0)
            activity_weight = self.get_activity_weight(activity, daily_operation_count)
            if activity and activity_weight > 0:
                row_weights[activity] = activity_weight

            # 提取文件类型权重（只提取文件类别）
            file_name = row.get("filename", "")
            for ext in self.file_type_weights.keys():
                if file_name.lower().endswith(ext):
                    row_weights[ext] = self.file_type_weights[ext]
                    break

            # 提取内容关键词权重（逐一关键词提取）
            content = row.get("content", "")
            if isinstance(content, str) and content.strip():
                content_keywords = content.lower().split()
                for keyword in content_keywords:
                    keyword_weight = self.content_weights.get(keyword, 0)
                    if keyword_weight > 0:
                        row_weights[keyword] = keyword_weight

            # 提取时间特征权重
            time_feature = row.get("Time_Feature", "")
            time_weight = self.get_time_feature_weight(time_feature)
            if time_feature and time_weight > 0:
                row_weights[time_feature] = time_weight

            # 将当前行的关键词权重加入结果
            extracted_keywords.append(row_weights)

        return extracted_keywords