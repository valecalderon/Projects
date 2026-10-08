using RentalQA.Core;
using RentalQA.Core.Models;
using RentalQA.Core.Repositories;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Registered as a singleton so data survives across requests during manual/API testing.
// Swap this one line for a SQL-Server-backed implementation in production.
builder.Services.AddSingleton<IRentalRepository, InMemoryRentalRepository>();
builder.Services.AddSingleton<RentalPriceCalculator>();

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI();

app.MapPost("/rentals", (CreateRentalRequest request, IRentalRepository repo, RentalPriceCalculator calculator) =>
{
    if (request.Days < 0 || request.DailyRate < 0)
        return Results.BadRequest("Days and daily rate must be non-negative.");

    var record = new RentalRecord
    {
        CustomerName = request.CustomerName,
        Days = request.Days,
        DailyRate = request.DailyRate,
        MilesDriven = request.MilesDriven,
        MilesIncluded = request.MilesIncluded,
        DueDate = request.DueDate,
        ReturnDate = request.ReturnDate
    };

    var saved = repo.Add(record);

    var total = calculator.CalculateTotalPrice(
        saved.Days, saved.DailyRate, saved.MilesDriven,
        saved.MilesIncluded, saved.DueDate, saved.ReturnDate);

    return Results.Created($"/rentals/{saved.Id}", new RentalResponse(saved, total));
});

app.MapGet("/rentals/{id:int}", (int id, IRentalRepository repo, RentalPriceCalculator calculator) =>
{
    var record = repo.GetById(id);
    if (record is null)
        return Results.NotFound();

    var total = calculator.CalculateTotalPrice(
        record.Days, record.DailyRate, record.MilesDriven,
        record.MilesIncluded, record.DueDate, record.ReturnDate);

    return Results.Ok(new RentalResponse(record, total));
});

app.MapGet("/rentals", (IRentalRepository repo) => Results.Ok(repo.GetAll()));

app.Run();

// DTOs kept in this file for simplicity - split into their own files as the API grows.
public record CreateRentalRequest(
    string CustomerName,
    int Days,
    decimal DailyRate,
    int MilesDriven,
    int MilesIncluded,
    DateTime DueDate,
    DateTime ReturnDate);

public record RentalResponse(RentalRecord Record, decimal TotalPrice);

// Required so the test project's WebApplicationFactory<Program> can see this class.
public partial class Program { }
