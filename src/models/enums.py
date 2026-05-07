import enum


class StatusEnum(str, enum.Enum):
    active = "active"
    inactive = "inactive"
    maintenance = "maintenance"


class SeverityEnum(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class MaintenanceTypeEnum(str, enum.Enum):
    preventive = "preventive"
    corrective = "corrective"
    predictive = "predictive"


class UserRoleEnum(str, enum.Enum):
    admin = "admin"
    technician = "technician"
    client = "client"
