namespace RentalQA.Core.Models;

public class RentalRecord
{
    public int Id { get; set; }
    public string CustomerName { get; set; } = string.Empty;
    public int Days { get; set; }
    public decimal DailyRate { get; set; }
    public int MilesDriven { get; set; }
    public int MilesIncluded { get; set; }
    public DateTime DueDate { get; set; }
    public DateTime ReturnDate { get; set; }
}
