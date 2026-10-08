using System.Net;
using System.Net.Http.Json;
using FluentAssertions;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace RentalQA.Tests;

// WebApplicationFactory spins up the real API in-memory, so these tests hit
// actual HTTP endpoints - this is what "API testing" means in practice,
// as opposed to unit tests that call C# methods directly.
public class RentalsApiTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public RentalsApiTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task PostRental_ValidRequest_ReturnsCreatedWithCorrectTotal()
    {
        var request = new
        {
            CustomerName = "Jane Doe",
            Days = 3,
            DailyRate = 29.99m,
            MilesDriven = 120,
            MilesIncluded = 100,
            DueDate = new DateTime(2026, 1, 10),
            ReturnDate = new DateTime(2026, 1, 10)
        };

        var response = await _client.PostAsJsonAsync("/rentals", request);

        response.StatusCode.Should().Be(HttpStatusCode.Created);

        var body = await response.Content.ReadFromJsonAsync<RentalResponse>();
        body.Should().NotBeNull();
        // 3 days * 29.99 = 89.97, plus 20 miles over at 0.79 = 15.80 -> 105.77
        body!.TotalPrice.Should().Be(105.77m);
    }

    [Fact]
    public async Task PostRental_NegativeDays_ReturnsBadRequest()
    {
        // API-layer validation test: confirms bad input is rejected before it
        // ever reaches the pricing logic or gets persisted.
        var request = new
        {
            CustomerName = "Test Customer",
            Days = -1,
            DailyRate = 20.00m,
            MilesDriven = 0,
            MilesIncluded = 100,
            DueDate = DateTime.Today,
            ReturnDate = DateTime.Today
        };

        var response = await _client.PostAsJsonAsync("/rentals", request);
        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    public async Task GetRental_UnknownId_ReturnsNotFound()
    {
        var response = await _client.GetAsync("/rentals/99999");
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    public async Task PostThenGetRental_DataPersistsCorrectly()
    {
        // This is the "backend" verification: after creating a record through the API,
        // fetch it back independently and confirm the STORED data matches what was sent -
        // not just that the API echoed the same object back to us in the same request.
        var request = new
        {
            CustomerName = "Backend Check",
            Days = 2,
            DailyRate = 40.00m,
            MilesDriven = 50,
            MilesIncluded = 100,
            DueDate = DateTime.Today,
            ReturnDate = DateTime.Today
        };

        var postResponse = await _client.PostAsJsonAsync("/rentals", request);
        var created = await postResponse.Content.ReadFromJsonAsync<RentalResponse>();

        var getResponse = await _client.GetAsync($"/rentals/{created!.Record.Id}");
        getResponse.StatusCode.Should().Be(HttpStatusCode.OK);

        var fetched = await getResponse.Content.ReadFromJsonAsync<RentalResponse>();
        fetched!.Record.CustomerName.Should().Be("Backend Check");
        fetched.TotalPrice.Should().Be(80.00m);
    }
}
