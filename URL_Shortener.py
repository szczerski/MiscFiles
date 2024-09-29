from flask import Flask, request, redirect, render_template, session
import random
import string
import os
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
from postgrest.exceptions import APIError

load_dotenv()

if os.environ.get("FLASK_ENV") == "development":
    with open(".env") as f:
        for line in f:
            var, val = line.strip().split("=")
            if var == "SUPABASE_URL":
                url = val
            elif var == "SUPABASE_KEY":
                key = val

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")
app.config["STATIC_FOLDER"] = "static"


url_mapping: dict[str, str] = {}
charset = string.ascii_lowercase + string.digits
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

last_short_url = ""
last_long_url = ""


def find_optimal_k(number_rows):
    chars = len(charset)
    k = 1
    max_ck = chars**k
    while max_ck <= number_rows:
        k += 1
        max_ck = chars**k

    return k


def create_short_url_entry(short_url, long_url):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        supabase.table("shortened_urls").insert(
            {
                "url_short": short_url,
                "url_long": long_url,
                "date": now,
                "clicked": 0,
                "created": 0,
            }
        ).execute()
    except APIError as e:
        raise e


@app.post("/")
def create_short_url():
    long_url = request.form.get("url")
    session["your_long_url"] = long_url
    if long_url:
        if not (long_url.startswith("http://") or long_url.startswith("https://")):
            if long_url.startswith("www."):
                long_url = "https://" + long_url
            else:
                long_url = "https://www." + long_url
        elif long_url.startswith("http://"):
            if long_url[7:].startswith("www.") or long_url[7:].find('/') != -1:
                long_url = "https://" + long_url[7:]
            else:
                long_url = "https://www." + long_url[7:]
        elif long_url.startswith("https://") and not long_url.startswith("https://www."):
            if long_url[8:].startswith("www.") or long_url[8:].find('/') != -1:
                long_url = "https://" + long_url[8:]
            else:
                long_url = "https://www." + long_url[8:]


    if long_url:
        existing_entry = (
            supabase.table("shortened_urls")
            .select("url_short")
            .eq("url_long", long_url)
            .execute()
        )

        if existing_entry.data:
            short_url = existing_entry.data[0]["url_short"]
            session["your_short_url"] = short_url
            response = (
                supabase.table("shortened_urls")
                .select("created")
                .eq("url_short", short_url)
                .execute()
            )
            current_created = response.data[0]["created"] if response.data else 0
            new_created = current_created + 1
            supabase.table("shortened_urls").update({"created": new_created}).eq(
                "url_short", short_url
            ).execute()

            return redirect("/")
        else:
            number_entry = supabase.table("shortened_urls").select("count").execute()
            number_of_rows = number_entry.data[0]["count"]
            optimal_k = find_optimal_k(number_of_rows)
            while True:

                short_url = "".join(random.choices(charset, k=optimal_k))
                session["your_short_url"] = short_url
                existing_short_url = (
                    supabase.table("shortened_urls")
                    .select("url_short")
                    .eq("url_short", short_url)
                    .execute()
                )
                if not existing_short_url.data:
                    create_short_url_entry(short_url, long_url)
                    return redirect("/")
    else:
        return "URL not provided", 400


@app.get("/")
def index():
    last_10_entries = (
        supabase.table("shortened_urls")
        .select("url_short", "url_long")
        .order("date", desc=True)
        .limit(10)
        .execute()
        .data
    )

    most_clicked_entries = (
        supabase.table("shortened_urls")
        .select("url_short", "url_long", "clicked")
        .order("clicked", desc=True)
        .limit(10)
        .execute()
        .data
    )

    most_created_entries = (
        supabase.table("shortened_urls")
        .select("url_short", "url_long", "created")
        .order("created", desc=True)
        .limit(10)
        .execute()
        .data
    )


    your_short_url = session.pop("your_short_url", None)
    your_long_url = session.pop("your_long_url", None)

    return render_template(
        "index.html",
        most_created_entries=most_created_entries,
        most_clicked_entries=most_clicked_entries,
        last_10_entries=last_10_entries,
        your_short_url=your_short_url,
        your_long_url=your_long_url,
    )


@app.get("/<short_url>")
def redirect_to_url(short_url):
    response = (
        supabase.table("shortened_urls")
        .select("url_long")
        .eq("url_short", short_url)
        .execute()
    )
    long_url = response.data[0]["url_long"] if response.data else None

    if long_url:
        response = (
            supabase.table("shortened_urls")
            .select("clicked")
            .eq("url_short", short_url)
            .execute()
        )
        current_clicked = response.data[0]["clicked"] if response.data else 0
        new_clicked = current_clicked + 1
        supabase.table("shortened_urls").update({"clicked": new_clicked}).eq(
            "url_short", short_url
        ).execute()

        return redirect(long_url)
    else:
        return "URL not found", 404


if __name__ == "__main__":
    app.run(debug=True)
